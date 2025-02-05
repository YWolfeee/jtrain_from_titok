#!/bin/bash

#SBATCH --account=dir_cosmos_misc
#SBATCH --partition=interactive
#SBATCH --container-mounts=/lustre/fsw/portfolios/dir/users/haotiany/joint_training/:/joint_training
#SBATCH --container-image=/lustre/fsw/portfolios/dir/users/haotiany/docker_images/imaginaire4_v9.2.2.sqsh
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00

nvidia-smi
cd /joint_training/jtrain_from_titok
pwd
source ~/.bashrc
pip install torchinfo

config_name='dry_run'
model_type="mlp"
tag="test_annealing"
ngpus=1
export PYTHONPATH=$(pwd)

export PYTHONPATH=$(pwd)
accelerate launch \
    --num_machines=1 --num_processes=1 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="TEMP_QY" \
    experiment.name="${config_name}_${tag}" \
    experiment.output_dir="${config_name}_${tag}" \
    model.use_reconstruction_regularization=True \
    model.reconstruction_regularization.name='matryoshka' \
    model.reconstruction_regularization.mask_ratio_method='hierarchical' \
    model.reconstruction_regularization.max_mask_rate=0.95 \
    \
    model.reconstruction_regularization.use_annealing=False \
    model.reconstruction_regularization.annealing.time_start=0.0 \
    model.reconstruction_regularization.annealing.time_end=0.1 \
    model.reconstruction_regularization.annealing.is_increasing=False \
    \
    model.reconstruction_regularization.use_policy=True \
    model.reconstruction_regularization.policy.use_advantage=True \
    model.reconstruction_regularization.policy.rate_weight=0.1 \
    model.reconstruction_regularization.policy.model_type=${model_type} \
    model.reconstruction_regularization.policy.num_heads=4 \
    model.reconstruction_regularization.policy.hidden_size=128 \
    \
    model.reconstruction_regularization.policy.annealing.use_annealing=True \
    model.reconstruction_regularization.policy.annealing.alpha_start=0.0 \
    model.reconstruction_regularization.policy.annealing.alpha_end=1.0 \
    \
    training.per_gpu_batch_size=32 \
    optimizer.params.learning_rate=4e-4 \
    training.max_train_steps=250_000 \
    losses.use_self_distilliation=False \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000252}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000009}.tar" \