"""Training script for TiTok.

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

Reference:
    https://github.com/huggingface/open-muse
"""
import math
import os
from pathlib import Path
import json
from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    get_config, create_dataloader)

from vdvae.hps import Hyperparams
from vdvae.vae import VAE
from vdvae.data import set_up_imagenet64_preprocess_func

def init_vae_settings():
    cifar10 = Hyperparams()
    cifar10.width = 384
    cifar10.lr = 0.0002
    cifar10.zdim = 16
    cifar10.wd = 0.01
    cifar10.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
    cifar10.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
    cifar10.warmup_iters = 100
    cifar10.dataset = 'cifar10'
    cifar10.n_batch = 16
    cifar10.ema_rate = 0.9999

    i32 = Hyperparams()
    i32.update(cifar10)
    i32.dataset = 'imagenet32'
    i32.ema_rate = 0.999
    i32.dec_blocks = "1x2,4m1,4x4,8m4,8x9,16m8,16x19,32m16,32x40"
    i32.enc_blocks = "32x15,32d2,16x9,16d2,8x8,8d2,4x6,4d4,1x6"
    i32.width = 512
    i32.n_batch = 8
    i32.lr = 0.00015
    i32.grad_clip = 200.
    i32.skip_threshold = 300.
    i32.epochs_per_eval = 1
    i32.epochs_per_eval_save = 1

    i64 = Hyperparams()
    i64.update(i32)
    i64.n_batch = 4
    i64.grad_clip = 220.0
    i64.skip_threshold = 380.0
    i64.dataset = 'imagenet64'
    i64.dec_blocks = "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12"
    i64.enc_blocks = "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5"
    
    i64.bottleneck_multiple = 0.25
    i64.no_bias_above = 64
    i64.num_mixtures = 10
    return i64

def compute_averages(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_elbo, total_distortion, total_rate = 0.0, 0.0, 0.0
    count = len(data)
    
    for key, values in data.items():
        total_elbo += values.get("elbo", 0.0)
        total_distortion += values.get("distortion", 0.0)
        total_rate += values.get("rate", 0.0)
    
    averages = {
        "average_elbo": total_elbo / count if count else 0.0,
        "average_distortion": total_distortion / count if count else 0.0,
        "average_rate": total_rate / count if count else 0.0
    }
    
    return averages

def main():
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="TiTok", log_level="INFO",
    output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.experiment.project,
            config=OmegaConf.to_container(config, resolve=True),
            init_kwargs={
                "wandb": {
                    "entity": "pixel-based-LM",
                    "name": config.experiment.name,
                    "id": config.experiment.name,
                }
            })
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")
        if config.training.enable_wandb:
            accelerator.get_tracker(tracker).log(
                OmegaConf.to_container(config, resolve=True))

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)
        
    accelerator.wait_for_everyone()

    H = init_vae_settings()
    H, preprocess_fn = set_up_imagenet64_preprocess_func(H)
    model = VAE(H)
    state_dict = torch.load('imagenet64-iter-1600000-model.th', map_location='cpu')
    new_state_dict = {}
    prefix_len = len('module.')
    for k in state_dict:
        if k.startswith('module.'):
            new_state_dict[k[prefix_len:]] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    state_dict = new_state_dict
    model.load_state_dict(state_dict)

    train_dataloader, eval_dataloader, train_eval_dataloader = create_dataloader(config, logger, accelerator)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    # model, train_dataloader, eval_dataloader, train_eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader, train_eval_dataloader)
    model = accelerator.prepare(model)

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum)

    # Start training.
    logger.info("***** Start VAE Inference *****")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    logger.info(f" Num batches = {num_batches}")
    
    stats_dict = {}
    stats_file = os.path.join(output_dir, "vae_results.json")
    
    # Load existing stats if file exists
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            stats_dict = json.load(f)

    logger.info("Successfully loaded stats with length", len(list(stats_dict.keys())))
    
    # One-time forward pass to get the stats
    for i, batch in enumerate(train_dataloader):
        
        model.eval()
        # Get filenames from batch
        fnames = batch['__key__']
        # Skip batch if all filenames already processed
        if all(fname in stats_dict for fname in fnames):
            logger.info(f"Processed {i} batches, Skipped")
            continue
        else:
            logger.info(f"Start processing {i} batches")
                
        # Get vae_input from batch
        vae_input = batch['vae_input'].permute(0, 2, 3, 1).contiguous().to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        inp, out = preprocess_fn(vae_input)
        with torch.no_grad():
            stats = model.forward(inp, out)
        
        # Gather stats and filenames from all processes
        gathered_fnames = fnames
        gathered_stats = accelerator.gather(stats)
        
        # Only save stats on main process
        if accelerator.is_main_process:
            # Save stats for each sample
            for j, fname in enumerate(gathered_fnames):
                sample_stats = {}
                for key in stats:
                    if torch.isnan(gathered_stats[key][j]) or torch.isinf(gathered_stats[key][j]):
                        logger.warning(f"Found unstable value for {fname} {key}: {gathered_stats[key][j]}, set to 0.0 instead")
                        sample_stats[key] = 0.0
                    else:
                        sample_stats[key] = gathered_stats[key][j].item()
                stats_dict[fname] = sample_stats
                
            if i % 100 == 0:
                logger.info(f"Gather stats: {gathered_stats}")
                # Save stats after each 100 batch
                with open(stats_file, "w") as f:
                    json.dump(stats_dict, f)
        
    # Save stats after each batch
    with open(stats_file, "w") as f:
        json.dump(stats_dict, f)
    
    logger.info("***** End VAE Inference *****")

    accelerator.wait_for_everyone()
    accelerator.end_training()
    avg_stats = compute_averages(stats_file)
    logger.info(f"Average INFO: {avg_stats}")


if __name__ == "__main__":
    main()