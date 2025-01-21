#!/bin/bash

#SBATCH --account=dir_cosmos_misc
#SBATCH --partition=interactive
#SBATCH --container-mounts=/lustre/fsw/portfolios/dir/users/haotiany/joint_training/:/joint_training
#SBATCH --container-image=/lustre/fsw/portfolios/dir/users/haotiany/docker_images/imaginaire4_v9.2.2.sqsh
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00

nvidia-smi
cd /joint_training/jtrain_from_titok
pwd
source ~/.bashrc
pip install torchinfo

config_name='titok_l256_4096_12'
ngpus=4
export PYTHONPATH=$(pwd)
accelerate launch \
    --num_machines=1 --num_processes=${ngpus} --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="temp" \
    experiment.name="${config_name}_hier" \
    experiment.output_dir="temp/${config_name}_stage1_run1" \
    model.use_reconstruction_regularization=True \
    model.reconstruction_regularization.name='matryoshka' \
    model.reconstruction_regularization.mask_ratio_method='hierarchical' \
    model.reconstruction_regularization.max_mask_rate=0.95 \
    model.reconstruction_regularization.use_annealing=True \
    model.reconstruction_regularization.annealing.time_start=0.25 \
    model.reconstruction_regularization.annealing.time_end=0.75 \
    model.reconstruction_regularization.annealing.is_increasing=False \
    training.per_gpu_batch_size=64 \
    optimizer.params.learning_rate=2e-4 \
    training.max_train_steps=250_000 \
    dataset.params.train_shards_path_or_url='datasets/imagenet-train-{000000..000252}.tar' \
    dataset.params.eval_shards_path_or_url='datasets/imagenet-val-{000000..000009}.tar' \
    losses.use_self_distilliation=False \

