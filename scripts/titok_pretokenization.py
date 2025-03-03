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
import itertools
from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    get_config, create_dataloader)

from modeling.titok import TiTok

def compute_positional_frequency(stats_dict):
    """Compute frequency of each code at each position from stats_dict.
    
    Args:
        stats_dict: List of dicts containing fname and full_tokens
        
    Returns:
        pos_freq: Dict mapping position -> code -> frequency
    """
    # Initialize position frequency dict
    pos_freq = {}
    
    # Iterate through each sample
    for sample in stats_dict:
        tokens = sample["full_tokens"]
        
        # Count frequency at each position
        for pos, code in enumerate(tokens):
            if pos not in pos_freq:
                pos_freq[pos] = {}
            
            code_int = int(code)
            if code_int not in pos_freq[pos]:
                pos_freq[pos][code_int] = 0
            pos_freq[pos][code_int] += 1
            
    # Convert counts to frequencies
    for pos in pos_freq:
        total = sum(pos_freq[pos].values())
        for code in pos_freq[pos]:
            pos_freq[pos][code] /= total
            
    return pos_freq

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

    model = TiTok(config)
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
    model.eval()

    # train_dataset, eval_dataset, train_eval_dataset = create_dataloader(config, logger, accelerator, return_dataset=True)
    
    # eval_dataloader = torch.utils.data.DataLoader(
    #     eval_dataset, 
    #     batch_size=config.training.per_gpu_batch_size,
    #     shuffle=False,
    #     num_workers=config.dataset.params.num_workers_per_gpu,
    #     pin_memory=True,
    #     persistent_workers=True
    # )
    
    # train_eval_dataloader = torch.utils.data.DataLoader(
    #     train_eval_dataset,
    #     batch_size=config.training.per_gpu_batch_size, 
    #     shuffle=False,
    #     num_workers=config.dataset.params.num_workers_per_gpu,
    #     pin_memory=True,
    #     persistent_workers=True
    # )

    _, eval_dataloader, train_eval_dataloader = create_dataloader(config, logger, accelerator)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, eval_dataloader, train_eval_dataloader = accelerator.prepare(model, eval_dataloader, train_eval_dataloader)
    # model = accelerator.prepare(model)

    total_batch_size_without_accum = config.training.per_gpu_batch_size # in this script, per_gpu_batch_size is simply the total gpu batch size
    try:
        if config.dataset_split == "val":
            iter_dataloader = eval_dataloader
            num_batches = math.ceil(50000 / total_batch_size_without_accum)
        else:
            iter_dataloader = train_eval_dataloader
            num_batches = math.ceil(
                config.experiment.max_train_examples / total_batch_size_without_accum)
    except:
        iter_dataloader = train_eval_dataloader
        num_batches = math.ceil(
            config.experiment.max_train_examples / total_batch_size_without_accum)

    # Start training.
    logger.info("***** Start TiTok Inference *****")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    logger.info(f" Num batches = {num_batches}")
    
    stats_dict = []  # Changed to list since we're appending
    stats_file = os.path.join(output_dir, "code_results.jsonl")
    freq_file = os.path.join(output_dir, "positional_frequencies.jsonl")
    
    # Load existing stats if file exists
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            for line in f:
                stats_dict.append(json.loads(line))
        logger.info(f"Successfully loaded stats with length {len(stats_dict)}")

    model.eval()
    
    for i, batch in enumerate(iter_dataloader):
        if i < config.start_batch:
            logger.info(f"Skipping {i} batches before indicated start")
            continue
        
        # Get filenames from batch
        fnames = batch['__key__']
        print(fnames)
        
        # Skip batch if all filenames already processed
        if all(fname in [s["fname"] for s in stats_dict] for fname in fnames):
            logger.info(f"Processed {i} batches, Skipped")
            continue
        else:
            logger.info(f"Start processing {i} batches")
                
        # Get vae_input from batch
        if "image" in batch:
            fnames = batch['__key__']
            # Get local rank and batch size
            local_rank = accelerator.local_process_index
            batch_size = len(fnames) // accelerator.num_processes
            
            # Calculate start and end indices for this process's portion
            start_idx = local_rank * batch_size
            end_idx = start_idx + batch_size
            
            # Get this process's portion of the batch
            
            images = batch["image"][start_idx:end_idx].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            dino_input = batch["dino_input"][start_idx:end_idx].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            vae_results = {k: v[start_idx:end_idx].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
                for k, v in batch["vae_results"].items()}
        
        local_model = accelerator.unwrap_model(model)
        with torch.no_grad():
            full_tokens, mask_rate = local_model.encode_tokens(images, dino_input=dino_input, vae_results=vae_results)
            logger.info(f"Local rank: {accelerator.local_process_index}", main_process_only=False)
            logger.info(f"First 10 tokens of first 5 items: {full_tokens[:5, :10]}", main_process_only=False)
            logger.info(f"Full tokens shape: {full_tokens.shape}", main_process_only=False)
            logger.info(f"First 5 mask rates: {mask_rate[:5]}", main_process_only=False)
            logger.info(f"Mask rate shape: {mask_rate.shape}", main_process_only=False)

        # gather results
        # Gather results across processes
        accelerator.wait_for_everyone()
        full_tokens = accelerator.gather(full_tokens)
        mask_rate = accelerator.gather(mask_rate)
        
        # Only save stats on main process
        if accelerator.is_main_process:
            # Save stats for each sample
            for j, fname in enumerate(fnames):
                sample = {"fname": fname, "full_tokens": full_tokens[j][:int(512*mask_rate[j])].cpu().numpy().tolist()}
                stats_dict.append(sample)
                # Write sample to jsonl file
                with open(stats_file, "a") as f:
                    f.write(json.dumps(sample) + "\n")
            
        if i % 100 == 0:
            logger.info(f"Stats_dict: {stats_dict}")
        
    # Compute and save positional frequencies
    if accelerator.is_main_process:
        pos_freq = compute_positional_frequency(stats_dict)
        with open(freq_file, "w") as f:
            json.dump(pos_freq, f)
    
    logger.info("***** End TiTok Inference *****")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()