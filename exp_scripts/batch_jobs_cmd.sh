#!/bin/bash

# List of YAML configuration files (manually specify here)
config_files=(
    # "titok_b128_4096_12.yaml"
    "titok_b256_4096_12.yaml"
    # "titok_l128_4096_12.yaml"
    "titok_l256_4096_12.yaml"
    # Add or remove YAML files here
)

# Parameters to iterate over for each YAML file
settings=(
    "use_reconstruction_regularization=False use_annealing=False use_self_distilliation=False"
    "use_reconstruction_regularization=True use_annealing=False use_self_distilliation=False"
    "use_reconstruction_regularization=True use_annealing=False use_self_distilliation=True"
    "use_reconstruction_regularization=True use_annealing=True use_self_distilliation=False"
    "use_reconstruction_regularization=True use_annealing=True use_self_distilliation=True"
)

# Define other parameters
per_gpu_batch_size=64
learning_rate=2e-4
is_increasing=True
output_root="results_try_hierarchical_loss"

# Loop through each YAML file
for config_file in "${config_files[@]}"; do
    # Extract the base name without extension for use in jobname
    config_name=$(basename "$config_file" .yaml)

    # Loop through each setting combination
    for setting in "${settings[@]}"; do
        # Evaluate the settings to create variables dynamically
        eval $setting


        # Dynamically create the job name
        jobname="${config_name}+use_ours=${use_reconstruction_regularization}+use_annealing=${use_annealing}+is_increasing=${is_increasing}+use_self_distilliation=${use_self_distilliation}"

        echo "jobname = $jobname"
        # Submit the job
        command="sbatch --job-name=$jobname --output='$output_root/$jobname/logs/slurm_%j.out' \
            exp_scripts/long_slurm.sh $config_name $per_gpu_batch_size $learning_rate $use_reconstruction_regularization $use_annealing $is_increasing $use_self_distilliation $output_root"
        echo "$command"
        eval "$command"
    done
done
