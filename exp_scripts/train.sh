#PBS -N titok_matryoshka
#PBS -S /bin/bash
#PBS -l select=1:ncpus=6:mem=45gb:ngpus=1:host=cvml11

config_name='dry_run'

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
WANDB_MODE=offline accelerate launch \
    --num_machines=1 --num_processes=1 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="${config_name}_stage1" \
    experiment.name="${config_name}_stage1_run1" \
    experiment.output_dir="${config_name}_stage1_run1" \
    model.use_reconstruction_regularization=True \
    model.reconstruction_regularization.name='matryoshka' \
    model.reconstruction_regularization.mask_ratio_method='uniform' \
    model.reconstruction_regularization.max_mask_rate=0.95 \
    model.reconstruction_regularization.use_annealing=True \
    model.reconstruction_regularization.annealing.time_start=0.25 \
    model.reconstruction_regularization.annealing.time_end=0.75 \
    model.reconstruction_regularization.annealing.is_increasing=False \
    training.per_gpu_batch_size=128 \
    optimizer.params.learning_rate=4e-4 \
    training.max_train_steps=250_000 \