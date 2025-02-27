#PBS -N zexp_pretokenize
#PBS -S /bin/bash
#PBS -l select=1:ncpus=12:mem=90gb:ngpus=1:host=cvml11

config_name='titok_b512_4096_12'
start_batch=0
dataset_split="val"
tag="pretokenize_after${start_batch}_${dataset_split}_cvml11"

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

export PYTHONPATH=$(pwd)
export WANDB_INIT_TIMEOUT=300

WANDB_MODE=offline accelerate launch \
    --num_machines=1 --num_processes=2 --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/titok_pretokenization.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="TEMP_QY" \
    experiment.name="${tag}" \
    experiment.output_dir="temp/${tag}" \
    experiment.init_weight="temp/stage1_checkpoint.bin" \
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
    model.reconstruction_regularization.policy.elbo.nll_only=True \
    model.reconstruction_regularization.policy.elbo.mean=0.5 \
    model.reconstruction_regularization.policy.elbo.lower=0.0 \
    model.reconstruction_regularization.policy.elbo.upper=1.0 \
    dataset.params.train_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-train-{000000..000320}.tar" \
    dataset.params.eval_shards_path_or_url="/mnt/rdata8/imagenet_wds/imagenet-val-{000000..000049}.tar" \
    dataset.params.num_workers_per_gpu=12 \
    training.per_gpu_batch_size=1024 \
    start_batch=${start_batch} \
    dataset_split=${dataset_split}
