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
import pprint

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    get_config, create_pretrained_tokenizer, 
    create_model_and_loss_module,
    create_dataloader,
    create_evaluator, auto_resume,
    eval_reconstruction_with_policy)


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

    if accelerator.local_process_index == 0:
        # download the maskgit-vq tokenizer weight
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="fun-research/TiTok", filename=f"{config.model.vq_model.pretrained_tokenizer_weight}", local_dir="./")
        
    accelerator.wait_for_everyone()

    pretrained_tokenizer = create_pretrained_tokenizer(config,
                                                       accelerator)

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type="titok")

    _, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)

    # Prepare everything with accelerator.
    logger.info("Preparing model and dataloaders")
    model, loss_module = accelerator.prepare(model, loss_module)
    if config.training.use_ema:
        ema_model.to(accelerator.device)
    model.to(accelerator.device)

    # Start Evaluation
    logger.info("***** Running Evaluation *****")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")

    logger.info("Computing metrics on the validation set.")
    eval_score = eval_reconstruction_with_policy(
        model,
        eval_dataloader,
        accelerator,
        evaluator,
        pretrained_tokenizer=pretrained_tokenizer,
        logger=logger
    )
    logger.info(
        f"EMA EVALUATION"
    )
    logger.info(
        "Compared to ground truth"
    )
    logger.info(pprint.pformat(eval_score))
    
    if accelerator.is_main_process:
        eval_log = {f'eval_policy_determined_tokens_vs_ground_truth/'+k: v for k, v in eval_score.items()}
        accelerator.log(eval_log)

    accelerator.wait_for_everyone()
    
    # mark as done for this running
    with open(os.path.join(output_dir, "done.txt"), "w") as f:
        f.write("\n")
    accelerator.end_training()

if __name__ == "__main__":
    main()