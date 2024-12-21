export PYTHONPATH=$(pwd)
WANDB_MODE=offline accelerate launch \
    --num_machines=1 --num_processes=4 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/titok_b64.yaml \
    experiment.project="titok_b64_stage1" \
    experiment.name="titok_b64_stage1_run1" \
    experiment.output_dir="titok_b64_stage1_run1" \
    training.per_gpu_batch_size=32