#!/bin/bash
###

# This bash file is used for the proposed new loss that
# explicitly trade-off between distortion and rate.
# We fix previous use_annealing, use_distill, is_increasing to False

###

config_name=$1 # 'titok_l256_4096_12'
per_gpu_batch_size=$2 # 64
lr=$3 # 2e-4
use_reconstruction_regularization=$4 # True
use_ours=$5
p_mean=$6
use_anneal=$7
use_encoder_mask=$8
output_root=$9
job_name=${10}         # Use as output dir
elbo_mode=${11}
wandb_projects=${12}
init_weight=${13}

ngpus=8

source ~/.bashrc
pip install torchinfo
which accelerate
cd /joint_training/jtrain_from_titok
export PYTHONPATH='/joint_training/jtrain_from_titok'


# if awk "BEGIN {exit !($alpha_start > 1)}"; then
#     use_annealing=False
#     alpha_start=0
# else
#     use_annealing=True
# fi
# echo "use_annealing=${use_annealing}, alpha_start=${alpha_start}"

accelerate launch \
    --num_machines=1 --num_processes=${ngpus} --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="${wandb_projects}" \
    experiment.name="${job_name}" \
    experiment.output_dir="${output_root}/${job_name}" \
    model.use_reconstruction_regularization=${use_reconstruction_regularization} \
    model.reconstruction_regularization.name='matryoshka' \
    model.reconstruction_regularization.mask_ratio_method='hierarchical' \
    model.reconstruction_regularization.max_mask_rate=0.95 \
    model.reconstruction_regularization.use_annealing=False \
    model.reconstruction_regularization.annealing.is_increasing=False \
    \
    model.vq_model.from_continuous=True \
    \
    model.reconstruction_regularization.use_policy=${use_ours} \
    model.reconstruction_regularization.policy.rate_weight=1.0 \
    model.reconstruction_regularization.policy.model_type='transformer' \
    model.reconstruction_regularization.policy.num_heads=4 \
    model.reconstruction_regularization.policy.hidden_size=128 \
    \
    model.reconstruction_regularization.policy.use_advantage=False \
    model.reconstruction_regularization.policy.use_pairwise=False \
    \
    model.reconstruction_regularization.policy.logit_head_type="gaussian_1" \
    \
    model.reconstruction_regularization.use_gumbel_softmax=False \
    model.reconstruction_regularization.gumbel_softmax.hard=True \
    model.reconstruction_regularization.gumbel_softmax.fix_tau=False \
    \
    model.reconstruction_regularization.use_encoder_mask=${use_encoder_mask} \
    model.reconstruction_regularization.policy.annealing.use_annealing=${use_anneal} \
    model.reconstruction_regularization.policy.annealing.alpha_start=0.0 \
    model.reconstruction_regularization.policy.annealing.alpha_end=0.5 \
    \
    model.reconstruction_regularization.policy.temperature.use_T=False \
    model.reconstruction_regularization.policy.temperature.T0=1000 \
    model.reconstruction_regularization.policy.temperature.alpha=1e-4 \
    \
    model.reconstruction_regularization.policy.gaussian_smoothing.use_gaussian_smoothing=False \
    model.reconstruction_regularization.policy.gaussian_smoothing.kernel_size=65 \
    \
    model.reconstruction_regularization.policy.elbo.nll_only=True \
    model.reconstruction_regularization.policy.elbo.elbo_mode=${elbo_mode} \
    model.reconstruction_regularization.policy.elbo.mean=${p_mean} \
    model.reconstruction_regularization.policy.elbo.start_mean=${p_mean} \
    model.reconstruction_regularization.policy.elbo.lower=0.0 \
    model.reconstruction_regularization.policy.elbo.upper=1.0 \
    \
    training.per_gpu_batch_size=${per_gpu_batch_size} \
    optimizer.params.learning_rate=${lr} \
    training.max_train_steps=500_000 \
    dataset.params.train_shards_path_or_url='datasets/imagenet-train-{000000..000252}.tar' \
    dataset.params.eval_shards_path_or_url='datasets/imagenet-val-{000000..000049}.tar' \
    experiment.init_weight=${init_weight}

    
