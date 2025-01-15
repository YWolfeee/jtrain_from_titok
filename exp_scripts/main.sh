config_name='titok_s128_4096_12'
ngpus=8

accelerate launch \
--num_machines=1 --num_processes=${ngpus} --machine_rank=0 \
--main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
scripts/train_titok.py config=configs/training/stage1/${config_name}.yaml \
experiment.project="${config_name}_stage1" \
experiment.name="${config_name}_stage1_run1" \
experiment.output_dir="${config_name}_stage1_run1" \
reconstruction_regularization.name='matryoshka' \
reconstruction_regularization.mask_ratio_method='uniform' \
reconstruction_regularization.max_mask_rate=0.95 \
reconstruction_regularization.annealing.time_start=0.25 \
reconstruction_regularization.annealing.time_end=0.75 \
reconstruction_regularization.annealing.is_increasing=False \
training.per_gpu_batch_size=128 \
optimizer.params.learning_rate=4e-4 \
training.max_train_steps=250_000 \

### if using batch size 32, modify the following parameters
# training.per_gpu_batch_size=32 \
# optimizer.params.learning_rate=1e-4 \
# training.max_train_steps=1_000_000 \

### if not using reconstruction regularization, simply delete the reconstruction_regularization section
### if not using annealing, simply delete the reconstruction_regularization.annealing section
