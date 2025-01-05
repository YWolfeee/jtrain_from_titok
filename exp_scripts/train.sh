#PBS -N titok_matryoshka
#PBS -S /bin/bash
#PBS -l select=1:ncpus=48:mem=360gb:ngpus=8:host=cvml11

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
accelerate launch \
    --num_machines=1 --num_processes=8 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/titok_s128_matryoshka_annealing.yaml \
    experiment.project="titok_s128_matryoshka_annealing_stage1" \
    experiment.name="titok_s128_matryoshka_annealing_stage1_run1" \
    experiment.output_dir="titok_s128_matryoshka_annealing_stage1_run1" \
    training.per_gpu_batch_size=32