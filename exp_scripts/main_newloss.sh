###

# This bash file is used for the proposed new loss that
# explicitly trade-off between distortion and rate.
# We fix previous use_annealing, use_distill, is_increasing to False

###

config_name=$1 # 'titok_l256_4096_12'
per_gpu_batch_size=$2 # 64
lr=$3 # 2e-4
use_reconstruction_regularization=$4 # True
use_policy_annealing=$5
rate_weight=$6
policy_network=$7
alpha_start=$8
output_root=$9
job_name=$10         # Use as output dir

ngpus=8

source ~/.bashrc
pip show torchinfo
which accelerate
cd /joint_training/jtrain_from_titok
export PYTHONPATH='/joint_training/jtrain_from_titok'

accelerate launch \
    --num_machines=1 --num_processes=${ngpus} --machine_rank=0 \
    --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
    scripts/train_titok.py config=configs/training/stage1/${config_name}.yaml \
    experiment.project="try_distortion_rate_tradeoff_loss" \
    experiment.name="${job_name}" \
    experiment.output_dir="${output_root}/${job_name}" \
    model.use_reconstruction_regularization=${use_reconstruction_regularization} \
    model.reconstruction_regularization.name='matryoshka' \
    model.reconstruction_regularization.mask_ratio_method='hierarchical' \
    model.reconstruction_regularization.max_mask_rate=0.95 \
    model.reconstruction_regularization.use_annealing=False \
    model.reconstruction_regularization.annealing.is_increasing=False \
    losses.use_self_distilliation=False \
    \
    model.reconstruction_regularization.use_policy=True \
    model.reconstruction_regularization.policy.use_advantage=True \
    model.reconstruction_regularization.policy.rate_weight=${rate_weight} \
    model.reconstruction_regularization.policy.model_type=${policy_network} \
    model.reconstruction_regularization.policy.num_heads=4 \
    model.reconstruction_regularization.policy.hidden_size=128 \
    \
    model.reconstruction_regularization.policy.annealing.use_annealing=${use_policy_annealing} \
    model.reconstruction_regularization.policy.annealing.alpha_start=${alpha_start} \
    model.reconstruction_regularization.policy.annealing.alpha_end=1.0 \
    \
    training.per_gpu_batch_size=${per_gpu_batch_size} \
    optimizer.params.learning_rate=${lr} \
    training.max_train_steps=250_000 \
    dataset.params.train_shards_path_or_url='datasets/imagenet-train-{000000..000252}.tar' \
    dataset.params.eval_shards_path_or_url='datasets/imagenet-val-{000000..000009}.tar' \
    

### if using batch size 32, modify the following parameters
# training.per_gpu_batch_size=32 \
# optimizer.params.learning_rate=1e-4 \
# training.max_train_steps=1_000_000 \

### if not using reconstruction regularization, simply delete the reconstruction_regularization section
### if not using annealing, simply delete the reconstruction_regularization.annealing section
