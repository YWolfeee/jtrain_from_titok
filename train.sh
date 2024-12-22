#PBS -N titok_matryoshka
#PBS -S /bin/bash
#PBS -l select=1:ncpus=52:mem=360gb:ngpus=4:host=cvml04

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
accelerate launch \
    --num_machines=1 --num_processes=4 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/titok_b64_matryoshka.yaml \
    experiment.project="titok_b64_stage1" \
    experiment.name="titok_b64_stage1_run1" \
    experiment.output_dir="titok_b64_stage1_run1" \
    training.per_gpu_batch_size=32