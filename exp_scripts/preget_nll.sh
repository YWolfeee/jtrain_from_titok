#PBS -N preget_nll
#PBS -S /bin/bash
#PBS -l select=1:ncpus=24:mem=180gb:ngpus=4:host=cvml10

config_name='titok_b128_4096_12'
tag="preget_nll_trial" # tag="preget_nll_trial_distributed_dataset_test"

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

WANDB_MODE=offline accelerate launch \
    --num_machines=1 --num_processes=4 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/preget_nll.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="TEMP_QY" \
    experiment.name="${tag}" \
    experiment.output_dir="temp/${tag}" \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000252}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000009}.tar" \
    training.per_gpu_batch_size=32