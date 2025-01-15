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

export PYTHONPATH=$(pwd)
accelerate launch \
    --num_machines=1 --num_processes=4 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/titok_s128_matryoshka_annealing.yaml \
    experiment.project="titok_s128_matryoshka_annealing_stage1" \
    experiment.name="titok_s128_matryoshka_annealing_stage1_run1" \
    experiment.output_dir="titok_s128_matryoshka_annealing_stage1_run1" \
    training.per_gpu_batch_size=32
