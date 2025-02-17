#!/bin/bash

#SBATCH --account=dir_cosmos_misc
#SBATCH --partition=batch
#SBATCH --container-mounts=/lustre/fsw/portfolios/dir/users/haotiany/joint_training/:/joint_training
#SBATCH --container-image=/lustre/fsw/portfolios/dir/users/haotiany/docker_images/imaginaire4_v9.2.2.sqsh
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00

nvidia-smi
cd /joint_training/jtrain_from_titok
pwd
source ~/.bashrc
pip install torchinfo

config_name='titok_b256_4096_12'
model_type="transformer"
tag="try_pairwise_from_scratch_beta=1_transformer_annealing"
ngpus=8
export PYTHONPATH=$(pwd)

# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
accelerate launch \
    --num_machines=1 --num_processes=${ngpus} --machine_rank=0 \
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
    model.reconstruction_regularization.policy.use_advantage=False \
    model.reconstruction_regularization.policy.rate_weight=1 \
    model.reconstruction_regularization.policy.use_pairwise=True \
    \
    model.reconstruction_regularization.use_gumbel_softmax=False \
    model.reconstruction_regularization.gumbel_softmax.hard=True \
    model.reconstruction_regularization.gumbel_softmax.fix_tau=False \
    \
    model.reconstruction_regularization.policy.annealing.use_annealing=True \
    model.reconstruction_regularization.policy.annealing.alpha_start=0.0 \
    model.reconstruction_regularization.policy.annealing.alpha_end=1.0 \
    model.reconstruction_regularization.policy.feature_extractor_name="facebook/dinov2-base" \
    model.reconstruction_regularization.policy.logit_head_type="categorical_8" \
    model.reconstruction_regularization.policy.gaussian_sampling.use_gaussian_sampling=False \
    model.reconstruction_regularization.policy.gaussian_sampling.sigma_0=2.0 \
    model.reconstruction_regularization.policy.gaussian_sampling.alpha=1e-3 \
    \
    model.reconstruction_regularization.policy.temperature.use_T=False \
    model.reconstruction_regularization.policy.temperature.T0=10000 \
    model.reconstruction_regularization.policy.temperature.alpha=1e-4 \
    \
    model.reconstruction_regularization.policy.gaussian_smoothing.use_gaussian_smoothing=True \
    model.reconstruction_regularization.policy.gaussian_smoothing.kernel_size=65 \
    \
    training.per_gpu_batch_size=64 \
    optimizer.params.learning_rate=4e-4 \
    training.max_train_steps=250_000 \
    dataset.params.train_shards_path_or_url='datasets/imagenet-train-{000000..000252}.tar' \
    dataset.params.eval_shards_path_or_url='datasets/imagenet-val-{000000..000009}.tar' \
    losses.use_self_distilliation=False

    # experiment.init_weight="titok_b256_4096_12+lr=4e-4+use_ours=True+use_annealing=False+is_increasing=True+use_self_distilliation=False.bin" \
    # \
    # dataset.params.train_shards_path_or_url='small_datasets/imagenet-train-000000.tar' \
    # dataset.params.eval_shards_path_or_url='small_datasets/imagenet-val-000000.tar' \
    # model.reconstruction_regularization.policy.training_regime.use_training_regime=True \
    # model.reconstruction_regularization.policy.training_regime.name='encoder_then_router_and_decoder' \
    # model.reconstruction_regularization.policy.training_regime.first_start=0.5 \
    # model.reconstruction_regularization.policy.training_regime.second_start=0.75 \
    # \
