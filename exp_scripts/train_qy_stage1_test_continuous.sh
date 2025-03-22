#PBS -N zexp_flextok_test
#PBS -S /bin/bash
#PBS -l select=1:ncpus=8:mem=90gb:ngpus=2:host=cvml11

config_name='titok_b128_4096_12'
model_type="transformer"
logit_head_type="gaussian_1"
rate_weight=0
mode=elastic
if [ "$mode" = "px" ]; then
    port=9999
elif [ "$mode" = "elastic" ]; then
    port=9980
else
    port=9990
fi
tag="try_stage1_flow_repa_correct_${mode}"

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

accelerate launch \
    --num_machines=1 --num_processes=2 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=${port} --same_network \
    scripts/train_titok.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="temp" \
    experiment.name="${tag}" \
    experiment.output_dir="temp/${tag}" \
    experiment.eval_every=1000 \
    \
    model.vq_model.from_continuous=True \
    \
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
    model.reconstruction_regularization.policy.rate_weight=0 \
    model.reconstruction_regularization.policy.use_pairwise=False \
    \
    model.reconstruction_regularization.use_gumbel_softmax=False \
    model.reconstruction_regularization.gumbel_softmax.hard=True \
    model.reconstruction_regularization.gumbel_softmax.fix_tau=False \
    \
    model.reconstruction_regularization.policy.annealing.use_annealing=True \
    model.reconstruction_regularization.policy.annealing.alpha_start=0.0 \
    model.reconstruction_regularization.policy.annealing.alpha_end=0.5 \
    model.reconstruction_regularization.policy.feature_extractor_name="facebook/dinov2-large" \
    model.reconstruction_regularization.policy.logit_head_type="gaussian_1" \
    \
    model.reconstruction_regularization.policy.elbo.nll_only=True \
    model.reconstruction_regularization.policy.elbo.elbo_mode="${mode}" \
    model.reconstruction_regularization.policy.elbo.start_mean=0.5 \
    model.reconstruction_regularization.policy.elbo.mean=0.5 \
    model.reconstruction_regularization.policy.elbo.lower=0.0 \
    model.reconstruction_regularization.policy.elbo.upper=1.0 \
    model.reconstruction_regularization.use_encoder_mask=True \
    \
    training.per_gpu_batch_size=64 \
    optimizer.params.learning_rate=5.62e-4 \
    lr_scheduler.params.warmup_steps=3814 \
    training.max_train_steps=500_000 \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000320}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000049}.tar" \