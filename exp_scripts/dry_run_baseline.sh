#PBS -N titok_matryoshka
#PBS -S /bin/bash
#PBS -l select=1:ncpus=24:mem=180gb:ngpus=4:host=cvml04
#PBS -e ~/logs/titok_matryoshka.%j.err
#PBS -o ~/logs/titok_matryoshka.%j.out

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
    experiment.project="${config_name}2_stage1" \
    experiment.name="${config_name}2_stage1_run1" \
    experiment.output_dir="${config_name}2_stage1_run1" \
    model.use_reconstruction_regularization=False \
    model.reconstruction_regularization.name='matryoshka' \
    model.reconstruction_regularization.mask_ratio_method='hierarchical' \
    model.reconstruction_regularization.max_mask_rate=0.95 \
    model.reconstruction_regularization.use_annealing=True \
    model.reconstruction_regularization.annealing.time_start=0.0 \
    model.reconstruction_regularization.annealing.time_end=0.1 \
    model.reconstruction_regularization.annealing.is_increasing=True \
    training.per_gpu_batch_size=16 \
    optimizer.params.learning_rate=4e-4 \
    training.max_train_steps=100_000 \
    losses.use_self_distilliation=False \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000252}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000009}.tar" \