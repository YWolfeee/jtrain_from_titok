"""Training utils for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""
import json
import os
import time
import math
from pathlib import Path
import pprint
import glob
from collections import defaultdict

from data import SimpleImageDataset, PretoeknizedDataSetJSONL
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel, ReconstructionLoss_Stage1, ReconstructionLoss_Stage2, MLMLoss, ARLoss
from modeling.titok import TiTok, PretrainedTokenizer
from modeling.maskgit import ImageBert, UViTBert
from modeling.rar import RAR
from evaluator import VQGANEvaluator
from demo_util import get_titok_tokenizer, sample_fn

from utils.viz_utils import make_viz_from_samples, make_viz_from_samples_generation
from torchinfo import summary


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_pretrained_tokenizer(config, accelerator=None):
    if config.model.vq_model.finetune_decoder:
        # No need of pretrained tokenizer at stage2
        pretrianed_tokenizer = None
    else:
        pretrianed_tokenizer = PretrainedTokenizer(config.model.vq_model.pretrained_tokenizer_weight)
        if accelerator is not None:
            pretrianed_tokenizer.to(accelerator.device)
    return pretrianed_tokenizer


def create_model_and_loss_module(config, logger, accelerator,
                                 model_type="titok"):
    """Creates TiTok model and loss module."""
    logger.info("Creating model and loss module.")
    if model_type == "titok":
        model_cls = TiTok
        loss_cls = ReconstructionLoss_Stage2 if config.model.vq_model.finetune_decoder else ReconstructionLoss_Stage1
    elif model_type == "maskgit":
        if config.model.generator.model_type == "ViT":
            model_cls = ImageBert
        elif config.model.generator.model_type == "UViT":
            model_cls == UViTBert
        else:
            raise ValueError(f"Unsupported generator model_type {config.model.generator.model_type}")
        loss_cls = MLMLoss
    elif model_type == "rar":
        model_cls = RAR
        loss_cls = ARLoss
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    model = model_cls(config)

    if config.experiment.get("init_weight", ""):
        # If loading a pretrained weight
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        if config.model.vq_model.finetune_decoder:
            # Add the MaskGIT-VQGAN's quantizer/decoder weight as well
            pretrained_tokenizer_weight = torch.load(
                config.model.vq_model.pretrained_tokenizer_weight, map_location="cpu"
            )
            # Only keep the quantize and decoder part
            pretrained_tokenizer_weight = {"pixel_" + k:v for k,v in pretrained_tokenizer_weight.items() if not "encoder." in k}
            model_weight.update(pretrained_tokenizer_weight)
        
        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    # Create loss module along with discrminator.
    loss_module = loss_cls(config=config)

    # Print Model for sanity check.
    if accelerator.is_main_process:
        if model_type in ["titok"]:
            input_size = (1, 3, config.dataset.preprocessing.crop_size, config.dataset.preprocessing.crop_size)
            model_summary_str = summary(model, input_size=input_size, depth=5,
            col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
            logger.info(model_summary_str)
        elif model_type in ["maskgit", "rar"]:
            input_size = (1, config.model.vq_model.num_latent_tokens)
            input_data = [
                torch.randint(0, config.model.vq_model.codebook_size, input_size),
                torch.ones(1, dtype=int)
            ]
            model_summary_str = summary(
                model, input_data=input_data, depth=7,
                col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
            logger.info(model_summary_str)
        else:
            raise NotImplementedError

    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module,
                     need_discrminator=True):
    """Creates optimizer for TiTok and discrminator."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Exclude terms we may not want to apply weight decay.
    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2)
    )

    if config.model.vq_model.finetune_decoder and need_discrminator:
        discriminator_learning_rate = optimizer_config.discriminator_learning_rate
        discriminator_named_parameters = list(loss_module.named_parameters())
        discriminator_gain_or_bias_params = [p for n, p in discriminator_named_parameters if exclude(n, p) and p.requires_grad]
        discriminator_rest_params = [p for n, p in discriminator_named_parameters if include(n, p) and p.requires_grad]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_gain_or_bias_params, "weight_decay": 0.},
                {"params": discriminator_rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        discriminator_optimizer = None

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """Creates learning rate scheduler for TiTok and discrminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=config.training.max_train_steps * accelerator.num_processes - config.losses.discriminator_start,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None
    return lr_scheduler, discriminator_lr_scheduler


def create_dataloader(config, logger, accelerator):
    """Creates data loader for training and testing."""
    logger.info("Creating dataloaders.")
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # TODO: add support on pre-tokenization dataset
    dataset = SimpleImageDataset(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=total_batch_size_without_accum,
        num_workers_per_gpu=dataset_config.num_workers_per_gpu,
        resize_shorter_edge=preproc_config.resize_shorter_edge,
        crop_size=preproc_config.crop_size,
        random_crop=preproc_config.random_crop,
        random_flip=preproc_config.random_flip,
    )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    train_eval_dataloader = dataset.train_eval_dataloader
    
    # potentially, use a pretokenized dataset for speed-up.
    if dataset_config.get("pretokenization", ""):
        train_dataloader = DataLoader(
            PretoeknizedDataSetJSONL(dataset_config.pretokenization),
            batch_size=config.training.per_gpu_batch_size,
            shuffle=True, drop_last=True, pin_memory=True)
        train_dataloader.num_batches = math.ceil(
            config.experiment.max_train_examples / total_batch_size_without_accum)
    
    return train_dataloader, eval_dataloader, train_eval_dataloader


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    evaluator = VQGANEvaluator(
        device=accelerator.device,
        enable_rfid=True,
        enable_inception_score=True,
        enable_codebook_usage_measure=True,
        enable_codebook_entropy_measure=True,
        num_codebook_entries=config.model.vq_model.codebook_size
    )
    return evaluator


def auto_resume(config, logger, accelerator, ema_model,
                num_update_steps_per_epoch, strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    # If resuming training.
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(
            config.experiment.output_dir, "checkpoint*")))
        logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
        if len(local_ckpt_list) >= 1:
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:
                checkpoint_paths = local_ckpt_list
            global_step = load_checkpoint(
                Path(checkpoint_paths[0]),
                accelerator,
                logger=logger,
                strict=strict
            )
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("Training from scratch.")
    return global_step, first_epoch


def train_one_epoch(config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    evaluators,
                    global_step,
                    pretrained_tokenizer=None):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    log_images = None
    for i, batch in enumerate(train_dataloader):
        model.train()
        if "image" in batch:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        else:
            raise ValueError(f"Not found valid keys: {batch.keys()}")

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        # Obtain proxy codes
        if pretrained_tokenizer is not None:
            pretrained_tokenizer.eval()
            proxy_codes = pretrained_tokenizer.encode(images)
        else:
            proxy_codes = None

        # QY: Update max mask rate based on annealing schedule if configured
        if config.model.reconstruction_regularization.use_annealing:
            max_mask_rate = get_titok_max_mask_rate(config, global_step)
            accelerator.unwrap_model(model).set_max_mask_rate(max_mask_rate)

        with accelerator.accumulate([model, loss_module]):
            reconstructed_images, extra_results_dict = model(images)
            # reconstructed_images.shape: [batch_size, 1024, H, W]
            if proxy_codes is None:
                autoencoder_loss, loss_dict = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step,
                    mode="generator",
                )
            elif config.losses.use_self_distilliation:
                # DEBUG: extra_results_dict should include "decode_mask_rate" and "self_distilliated_codes" in train_titok.py forward()
                current_decode_mask_rate = extra_results_dict["decode_mask_rate"]
                self_distillated_codes = extra_results_dict["self_distilliated_codes"]
                
                # DEBUG: the number of classes aligning with the codebook size provided in the loss 
                #       (VQGAN codebook size, as defined in losses.py ReconstructionLoss_Stage1.target_codebook_size)
                
                # DEBUG: self_distillated_codes currently is in the same shape as reconstructed_images [batch_size, 1024, H, W]
                #       Reshape it to [batch_size, 1024, H*W] for softmax operation and replacement operation
                self_distillated_codes = self_distillated_codes.contiguous()
                self_distillated_codes = self_distillated_codes.view(self_distillated_codes.shape[0], 1024, -1)
                
                # DEBUG: self_distillated_codes so far is still vector, to enable its computation for further cross-entropy loss
                #        We need to convert it to a probability distribution over the codebook size
                self_distillated_codes = self_distillated_codes.softmax(dim=1)

                # DEBUG: proxy_codes is in shape of [batch_size, H*W], where each proxy_codes[i][j] is the index of the codebook
                #       Originally, it will be fed as the y (hard label) in the later compuation of cross-entropy loss nn.CrossEntropyLoss()
                #       But self_distillated_codes is not index of codebook (but a prob distribution), so we need to convert proxy_codes to one-hot vector
                #       This is for the later stage of torch.where(operation)
                proxy_codes = torch.nn.functional.one_hot(proxy_codes, num_classes=1024).permute(0, 2, 1).to(self_distillated_codes.dtype)
                
                # DEBUG: current_decode_mask_rate is in shape of [batch_size, ]
                #       Expand current_decode_mask_rate to match dimensions [batch_size, 1024, H*W]
                current_decode_mask_rate = current_decode_mask_rate.view(-1, 1, 1)
                
                # current_decode_mask_rate: [batch_size, 1, 1], proxy_codes: [batch_size, 1024, H*W], self_distillated_codes: [batch_size, 1024, H*W]
                # DEBUG: This is to replace the self-distilliated codes with the proxy codes if the decode_mask_rate is 0 (< 1/16)
                self_distillated_codes = torch.where(current_decode_mask_rate - 1/16 < 0, proxy_codes, self_distillated_codes)
                
                # DEBUG: to handle different shape of code provided (index or distribution)
                #        we set mode="with_self_distilliation" for distribution, "with_ground_truth" for index
                autoencoder_loss, loss_dict = loss_module(
                    # DEBUG: The following comment is VERY IMPORTANT
                    self_distillated_codes, # DEBUG: change this to proxy_codes can work properly (loss can decrease), but self_distilliated_codes will make loss stuck
                    reconstructed_images,
                    extra_results_dict,
                    mode="with_self_distilliation"
                )
            else:
                autoencoder_loss, loss_dict = loss_module(
                    proxy_codes,
                    reconstructed_images,
                    extra_results_dict,
                    mode="with_ground_truth"
                )

            # Gather the losses across all processes for logging.
            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight"]:
                    if type(v) == torch.Tensor:
                        autoencoder_logs["train/" + k] = v.cpu().item()
                    else:
                        autoencoder_logs["train/" + k] = v
                else:
                    autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

            accelerator.backward(autoencoder_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            # Train discriminator.
            discriminator_logs = defaultdict(float)
            if config.model.vq_model.finetune_decoder and accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                discriminator_logs = defaultdict(float)
                discriminator_loss, loss_dict_discriminator = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step=global_step,
                    mode="discriminator",
                )

                # Gather the losses across all processes for logging.
                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                        discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(discriminator_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)

                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()
        
                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)
                
                discriminator_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                    f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                if log_images is None:
                    log_images = images[:config.training.num_generated_images]
                    log_fnames = fnames[:config.training.num_generated_images]
                reconstruct_images(
                    model,
                    log_images,
                    log_fnames,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                    pretrained_tokenizer=pretrained_tokenizer
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            if (global_step + 1) % config.experiment.eval_loss_every == 0:
                logger.info(f"Global step: {global_step + 1}")
                eval_loss_dict = eval_loss(
                    model,
                    train_dataloader,
                    accelerator,
                    loss_module,
                    pretrained_tokenizer=pretrained_tokenizer
                )
                logger.info(pprint.pformat(eval_loss_dict))
                eval_loss_log = {f'eval_loss/'+k: v for k, v in eval_loss_dict.items()}
                accelerator.log(eval_loss_log, step=global_step + 1)

            # Evaluate reconstruction.
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    # Eval for EMA.
                    decode_mask_rates = [0.0, 0.25, 0.5, 0.75]
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluators,
                        pretrained_tokenizer=pretrained_tokenizer
                    )
                    for i in range(4):
                        logger.info(
                            f"EMA EVALUATION with {(1 - decode_mask_rates[i]) * 100}% tokens"
                            f"Step: {global_step + 1} "
                        )
                        logger.info(
                            f"Compared to ground truth"
                        )
                        logger.info(pprint.pformat(eval_scores[i]))
                        
                        if accelerator.is_main_process:
                            eval_log = {f'ema_eval_{(1 - decode_mask_rates[i]) * 100}%_tokens_vs_ground_truth/'+k: v for k, v in eval_scores[i].items()}
                            accelerator.log(eval_log, step=global_step + 1)
                        
                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())
                else:
                    # Eval for non-EMA.
                    for i in range(4):
                        logger.info(
                            f"EVALUATION with {(1 - decode_mask_rates[i]) * 100}% tokens"
                            f"Step: {global_step + 1} "
                        )
                        logger.info(
                            f"Compared to ground truth"
                        )
                        logger.info(pprint.pformat(eval_scores[i]))
                        
                        if accelerator.is_main_process:
                            eval_log = {f'eval_{(1 - decode_mask_rates[i]) * 100}%_tokens_vs_ground_truth/'+k: v for k, v in eval_scores[i].items()}
                            accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step

def get_titok_max_mask_rate(config, global_step):
    """QY: Get the max mask rate for TiTok model."""
    annealing = config.model.reconstruction_regularization.annealing
    is_increasing = annealing.is_increasing
    time_start = annealing.time_start * config.training.max_train_steps
    time_end = annealing.time_end * config.training.max_train_steps
    alpha = (global_step - time_start) / (time_end - time_start)
    end_mask_rate = config.model.reconstruction_regularization.max_mask_rate if is_increasing else 0.0
    start_mask_rate = 0.0 if is_increasing else config.model.reconstruction_regularization.max_mask_rate
    if global_step < time_start:
        return start_mask_rate
    elif global_step > time_end:
        return end_mask_rate
    else:
        return alpha * end_mask_rate + (1 - alpha) * start_mask_rate

def get_rar_random_ratio(config, cur_step):
    randomness_anneal_start = config.model.generator.randomness_anneal_start
    randomness_anneal_end = config.model.generator.randomness_anneal_end
    if cur_step < randomness_anneal_start:
        return 1.0
    elif cur_step > randomness_anneal_end:
        return 0.0
    else:
        return 1.0 - (cur_step - randomness_anneal_start) / (randomness_anneal_end - randomness_anneal_start)

def train_one_epoch_generator(
                    config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer,
                    lr_scheduler,
                    train_dataloader,
                    tokenizer,
                    global_step,
                    model_type="maskgit"):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    for i, batch in enumerate(train_dataloader):
        model.train()
        if config.dataset.params.get("pretokenization", ""):
            # the data is already pre-tokenized
            conditions, input_tokens = batch
            input_tokens = input_tokens.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            conditions = conditions.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        else:
            # tokenize on the fly
            if "image" in batch:
                images = batch["image"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
                conditions = batch["class_id"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )

                # Encode images on the flight.
                with torch.no_grad():
                    tokenizer.eval()
                    input_tokens = tokenizer.encode(images)[1]["min_encoding_indices"].reshape(images.shape[0], -1)
            else:
                raise ValueError(f"Not found valid keys: {batch.keys()}")

        data_time_meter.update(time.time() - end)

        unwrap_model = accelerator.unwrap_model(model)


        if model_type == "maskgit":
            # Randomly masking out input tokens.
            masked_tokens, masks = unwrap_model.masking_input_tokens(
                input_tokens)
        elif model_type == "rar":
            unwrap_model.set_random_ratio(get_rar_random_ratio(config, global_step))
        else:
            raise NotImplementedError
            

        with accelerator.accumulate([model]):

            if model_type == "maskgit":
                logits = model(masked_tokens, conditions,
                            cond_drop_prob=config.model.generator.class_label_dropout)
                loss, loss_dict= loss_module(logits, input_tokens, weights=masks)
            elif model_type == "rar":
                condition = unwrap_model.preprocess_condition(
                    conditions, cond_drop_prob=config.model.generator.class_label_dropout
                )
                logits, labels = model(input_tokens, condition, return_labels=True)
                loss, loss_dict = loss_module(logits, labels)
            # Gather the losses across all processes for logging.
            gen_logs = {}
            for k, v in loss_dict.items():
                gen_logs["train/" + k] = accelerator.gather(v).mean().item()
            accelerator.backward(loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Loss: {gen_logs['train/loss']:0.4f} "
                    f"Accuracy: {gen_logs['train/correct_tokens']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(gen_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                generate_images(
                    model,
                    tokenizer,
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break

    return global_step

@torch.no_grad()
def eval_loss(
    model,
    eval_loader,
    accelerator,
    loss_module,
    pretrained_tokenizer=None,
    sampled_batches=4
):
    decode_mask_rates = [i / 16 for i in range(16)]
    local_model = accelerator.unwrap_model(model)
    model.eval()
    eval_loss_dict = {}
    t = 0
    for batch in eval_loader:
        if t >= sampled_batches:
            break
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        if pretrained_tokenizer is not None:
            pretrained_tokenizer.eval()
            proxy_codes = pretrained_tokenizer.encode(images)
        for i, decode_mask_rate in enumerate(decode_mask_rates):
            reconstructed_images, extra_results_dict = local_model(images, decode_mask_rate=decode_mask_rate)
            # compare with ground truth
            if proxy_codes is None:
                _, loss_dict = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    0, # ignore effect of global_step in this step
                    mode="generator",
                )
            else:
                _, loss_dict = loss_module(
                    proxy_codes,
                    reconstructed_images,
                    extra_results_dict,
                    mode="with_ground_truth"
                )
            current_key = f"{(1 - decode_mask_rates[i]) * 100}%_tokens_vs_ground_truth"
            if current_key not in eval_loss_dict:
                eval_loss_dict[current_key] = accelerator.gather(loss_dict["reconstruction_loss"]).mean().item()
            else:
                eval_loss_dict[current_key] += accelerator.gather(loss_dict["reconstruction_loss"]).mean().item()

            if i >= 1:
                if proxy_codes is None:
                    _, loss_dict = loss_module(
                        previous_reconstructed_images,
                        reconstructed_images,
                        extra_results_dict,
                        0, # ignore effect of global_step in this step
                        mode="generator",
                    )
                else:
                    previous_reconstructed_images = previous_reconstructed_images.contiguous()
                    previous_reconstructed_images = previous_reconstructed_images.view(previous_reconstructed_images.shape[0], 1024, -1)
                    previous_reconstructed_images = previous_reconstructed_images.softmax(dim=1)
                    _, loss_dict = loss_module(
                        previous_reconstructed_images,
                        reconstructed_images,
                        extra_results_dict,
                        mode="with_self_distilliation"
                    )
                current_key = f"{(1 - decode_mask_rates[i]) * 100}%_tokens_vs_{(1 - decode_mask_rates[i-1]) * 100}%_tokens"
                if current_key not in eval_loss_dict:
                    eval_loss_dict[current_key] = accelerator.gather(loss_dict["reconstruction_loss"]).mean().item()
                else:
                    eval_loss_dict[current_key] += accelerator.gather(loss_dict["reconstruction_loss"]).mean().item()
            
            previous_reconstructed_images = reconstructed_images
        t += 1

    for k, v in eval_loss_dict.items():
        eval_loss_dict[k] = v / sampled_batches

    model.train()
    return eval_loss_dict



@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    evaluators,
    pretrained_tokenizer=None
):
    model.eval()
    # There are totally 4 evalators:
    # 4 for [0.0, gt], [0.25, gt], [0.5, gt], [0.75, gt]

    if len(evaluators) != 4:
        raise ValueError(f"Evaluator length should be 4, but got {len(evaluators)}")
    
    for evaluator in evaluators:
        evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)

    decode_mask_rates = [0.0, 0.25, 0.5, 0.75]

    for batch in eval_loader:
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        images_lists = []
        original_images = torch.clone(images)
        original_images = torch.clamp(original_images, 0.0, 1.0)
        for decode_mask_rate in decode_mask_rates:
            reconstructed_images, model_dict = local_model(images, decode_mask_rate=decode_mask_rate)
            if pretrained_tokenizer is not None:
                reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
            reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
            # Quantize to uint8
            reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
            images_lists.append(reconstructed_images)
        
        for i in range(4):
            evaluators[i].update(original_images, images_lists[i].squeeze(2), model_dict["min_encoding_indices"])

    model.train()
    return [evaluator.result() for evaluator in evaluators]


@torch.no_grad()
def reconstruct_images(model, original_images, fnames, accelerator, 
                    global_step, output_dir, logger, config=None,
                    pretrained_tokenizer=None):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_images)
    reconstructed_images_list = []
    # QY: Eval with different decode_mask_rate
    mask_rate_list = [i / 16 for i in range(17)]
    for decode_mask_rate in mask_rate_list:
        reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens, decode_mask_rate=decode_mask_rate)
        if pretrained_tokenizer is not None:
            reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
        reconstructed_images_list.append(reconstructed_images)
    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images,
        reconstructed_images_list
    )
    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Reconstruction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_saving):
        filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
        path = os.path.join(root, filename)
        img.save(path)

    model.train()


@torch.no_grad()
def generate_images(model, tokenizer, accelerator, 
                    global_step, output_dir, logger, config=None):
    model.eval()
    tokenizer.eval()
    logger.info("Generating images...")
    generated_image = sample_fn(
        accelerator.unwrap_model(model),
        tokenizer,
        guidance_scale=config.model.generator.get("guidance_scale", 3.0),
        guidance_decay=config.model.generator.get("guidance_decay", "constant"),
        guidance_scale_pow=config.model.generator.get("guidance_scale_pow", 3.0),
        randomize_temperature=config.model.generator.get("randomize_temperature", 2.0),
        softmax_temperature_annealing=config.model.generator.get("softmax_temperature_annealing", False),
        num_sample_steps=config.model.generator.get("num_steps", 8),
        device=accelerator.device,
        return_tensor=True
    )
    images_for_saving, images_for_logging = make_viz_from_samples_generation(
        generated_image)

    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"Train Generated": [images_for_saving]}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Generated": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_generated_images"
    os.makedirs(root, exist_ok=True)
    filename = f"{global_step:08}_s-generated.png"
    path = os.path.join(root, filename)
    images_for_saving.save(path)

    model.train()
    return


def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # Delete oldest checkpoint if more than 3 exist
    if accelerator.is_main_process:
        checkpoints = sorted([d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")], 
                           key=lambda x: int(x.name.split("-")[1]))  # Sort by checkpoint number
        if len(checkpoints) >= 3:
            oldest_checkpoint = checkpoints[0]  # First one has lowest number
            import shutil
            shutil.rmtree(oldest_checkpoint)
            logger.info(f"Removed old checkpoint at {oldest_checkpoint}")

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    return save_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)
    
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)