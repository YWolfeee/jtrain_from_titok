#PBS -N zexp_ar_test
#PBS -S /bin/bash
#PBS -l select=1:ncpus=12:mem=90gb:ngpus=2:host=cvml11

config_name='rar'
tag="rar_test_0"

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

accelerate launch \
    --num_machines=4 --num_processes=32 --machine_rank=${MACHINE_RANK} \
    --main_process_ip=${ROOT_IP} --main_process_port=${ROOT_PORT} --same_network \
    scripts/train_rar.py config=configs/training/generator/rar.yaml \
    experiment.project="TEMP_QY" \
    experiment.name="${config_name}_${tag}" \
    experiment.output_dir="temp/${tag}" \
    model.generator.hidden_size=768 \
    model.generator.num_hidden_layers=24 \
    model.generator.num_attention_heads=16 \
    model.generator.intermediate_size=3072