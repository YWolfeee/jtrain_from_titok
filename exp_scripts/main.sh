config_name=$1 # 'titok_l256_4096_12'
per_gpu_batch_size=$2 # 64
lr=$3 # 2e-4
use_reconstruction_regularization=$4 # True
use_annealing=$5    #False
is_increasing=$6    #False
use_self_distilliation=$7
output_root=$8
job_name=$9         # Use as output dir

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
    experiment.project="try_hierarchical_loss" \
    experiment.name="${job_name}" \
    experiment.output_dir="${output_root}/${job_name}" \
    model.use_reconstruction_regularization=${use_reconstruction_regularization} \
    model.reconstruction_regularization.name='matryoshka' \
    model.reconstruction_regularization.mask_ratio_method='hierarchical' \
    model.reconstruction_regularization.max_mask_rate=0.95 \
    model.reconstruction_regularization.use_annealing=${use_annealing} \
    model.reconstruction_regularization.annealing.time_start=0.0 \
    model.reconstruction_regularization.annealing.time_end=0.1 \
    model.reconstruction_regularization.annealing.is_increasing=${is_increasing} \
    losses.use_self_distilliation=${use_self_distilliation} \
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
