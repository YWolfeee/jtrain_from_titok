#PBS -N titok_matryoshka
#PBS -S /bin/bash
#PBS -l select=1:ncpus=6:mem=45gb:ngpus=1:host=cvml11

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
WANDB_MODE=offline accelerate launch \
    --num_machines=1 --num_processes=1 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/dry_run.yaml \
    experiment.project='dry_run_stage1' \
    experiment.name='dry_run_stage1_run1' \
    experiment.output_dir='dry_run_stage1_run1' \
    reconstruction_regularization.name='matryoshka' \
    reconstruction_regularization.mask_ratio_method='uniform' \
    reconstruction_regularization.max_mask_rate=0.95 \
    reconstruction_regularization.annealing.time_start=0.25 \
    reconstruction_regularization.annealing.time_end=0.75 \
    reconstruction_regularization.annealing.is_increasing=False \
    training.per_gpu_batch_size=128 \
    optimizer.params.learning_rate=4e-4 \
    training.max_train_steps=250_000 \