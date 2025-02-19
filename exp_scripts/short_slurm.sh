#!/bin/bash

#SBATCH --account=dir_cosmos_misc
#SBATCH --partition=batch
#SBATCH --container-mounts=/lustre/fsw/portfolios/dir/users/haotiany/joint_training/:/joint_training
#SBATCH --container-image=/lustre/fsw/portfolios/dir/users/haotiany/docker_images/imaginaire4_v9.2.2.sqsh
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00

# nvidia-smi
cd /joint_training/jtrain_from_titok
pwd
source ~/.bashrc

config_name="titok_b256_4096_12"
model_type="transformer"
tag="debug_gaussian_nan"
ngpus=8
export PYTHONPATH=$(pwd)

# accelerate launch \
#     --num_machines=1 --num_processes=${ngpus} --machine_rank=0 \
#     --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    scripts/train_titok.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="TEMP_QY" \
    experiment.name="${config_name}_${tag}" \
    experiment.output_dir="temp/${tag}" \
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
    model.reconstruction_regularization.policy.rate_weight=1 \
    model.reconstruction_regularization.policy.use_pairwise=True \
    \
    model.reconstruction_regularization.use_gumbel_softmax=False \
    model.reconstruction_regularization.gumbel_softmax.hard=True \
    model.reconstruction_regularization.gumbel_softmax.fix_tau=False \
    \
    model.reconstruction_regularization.policy.annealing.use_annealing=False \
    model.reconstruction_regularization.policy.annealing.alpha_start=0.02 \
    model.reconstruction_regularization.policy.annealing.alpha_end=1.0 \
    model.reconstruction_regularization.policy.feature_extractor_name="facebook/dinov2-base" \
    model.reconstruction_regularization.policy.logit_head_type="gaussian_1" \
    \
    model.reconstruction_regularization.policy.temperature.use_T=False \
    model.reconstruction_regularization.policy.temperature.T0=10000 \
    model.reconstruction_regularization.policy.temperature.alpha=1e-4 \
    \
    model.reconstruction_regularization.policy.gaussian_smoothing.use_gaussian_smoothing=False \
    model.reconstruction_regularization.policy.gaussian_smoothing.kernel_size=65 \
    \
    training.per_gpu_batch_size=64 \
    optimizer.params.learning_rate=4e-4 \
    training.max_train_steps=250_000 \
    dataset.params.train_shards_path_or_url='datasets/imagenet-train-{000000..000252}.tar' \
    dataset.params.eval_shards_path_or_url='datasets/imagenet-val-{000000..000009}.tar' \
    
    # experiment.init_weight="results_try_new_design/gaussian+rate_weight=1+policy_network=transformer+anneal_policy+alpha_start=2/checkpoint-30000/unwrapped_model/pytorch_model.bin"

    # \
    # dataset.params.train_shards_path_or_url='small_datasets/imagenet-train-000000.tar' \
    # dataset.params.eval_shards_path_or_url='small_datasets/imagenet-val-000000.tar' \
