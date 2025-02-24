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
    model.generator.intermediate_size=3072 \
    model.vq_model.pretrained_tokenizer_name="ours" \
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
    model.reconstruction_regularization.policy.elbo.lower=0.2 \
    model.reconstruction_regularization.policy.elbo.upper=1.0 \