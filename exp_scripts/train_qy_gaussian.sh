#PBS -N titok_matryoshka
#PBS -S /bin/bash
#PBS -l select=1:ncpus=2:mem=90gb:ngpus=2:host=cvml05

config_name='titok_b64_4096_12'
model_type='mlp'
tag='gaussian'

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_machines=1 --num_processes=2 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="TEMP_QY" \
    experiment.name="${config_name}_${tag}" \
    experiment.output_dir="temp/${config_name}_${tag}" \
    model.use_reconstruction_regularization=True \
    model.reconstruction_regularization.name='matryoshka' \
    model.reconstruction_regularization.mask_ratio_method='hierarchical' \
    model.reconstruction_regularization.max_mask_rate=0.95 \
    \
    model.reconstruction_regularization.use_annealing=False \
    model.reconstruction_regularization.annealing.time_start=0.0 \
    model.reconstruction_regularization.annealing.time_end=0.1 \
    model.reconstruction_regularization.annealing.is_increasing=False \
    \
    model.reconstruction_regularization.use_policy=True \
    model.reconstruction_regularization.policy.model_type=${model_type} \
    model.reconstruction_regularization.policy.num_heads=4 \
    model.reconstruction_regularization.policy.hidden_size=128 \
    \
    model.reconstruction_regularization.policy.use_advantage=True \
    model.reconstruction_regularization.policy.rate_weight=0.001 \
    \
    model.reconstruction_regularization.use_gumbel_softmax=True \
    model.reconstruction_regularization.gumbel_softmax.hard=True \
    model.reconstruction_regularization.gumbel_softmax.fix_tau=True \
    \
    model.reconstruction_regularization.policy.annealing.use_annealing=False \
    model.reconstruction_regularization.policy.annealing.alpha_start=0.2 \
    model.reconstruction_regularization.policy.annealing.alpha_end=1.0 \
    \
    model.reconstruction_regularization.policy.temperature.use_T=True \
    model.reconstruction_regularization.policy.temperature.T0=10000 \
    model.reconstruction_regularization.policy.temperature.alpha=1e-4 \
    \
    model.reconstruction_regularization.policy.gaussian_smoothing.use_gaussian_smoothing=True \
    model.reconstruction_regularization.policy.gaussian_smoothing.kernel_size=65 \
    model.reconstruction_regularization.policy.gaussian_smoothing.start_time=0.0 \
    model.reconstruction_regularization.policy.gaussian_smoothing.end_time=1.0 \
    model.reconstruction_regularization.policy.gaussian_smoothing.start_value=30.0 \
    model.reconstruction_regularization.policy.gaussian_smoothing.end_value=0.1 \
    \
    training.per_gpu_batch_size=64 \
    optimizer.params.learning_rate=4e-4 \
    training.max_train_steps=250_000 \
    losses.use_self_distilliation=False \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000252}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000009}.tar" \