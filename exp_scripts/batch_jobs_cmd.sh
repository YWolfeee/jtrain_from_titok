#!/bin/bash

# List of YAML configuration files (manually specify here)
config_files=(
    # "titok_b128_4096_12.yaml"
    "titok_b256_4096_12.yaml"
    "titok_l128_4096_12.yaml"
    "titok_l256_4096_12.yaml"
    # Add or remove YAML files here
)

# Parameters to iterate over for each YAML file
settings=(
    "use_reconstruction_regularization=False use_annealing=False is_increasing=False"
    "use_reconstruction_regularization=True use_annealing=False is_increasing=False"
    "use_reconstruction_regularization=True use_annealing=True is_increasing=False"
    "use_reconstruction_regularization=True use_annealing=True is_increasing=True"
)

# Define other parameters
per_gpu_batch_size=64
learning_rate=2e-4
output_root="results"

# Loop through each YAML file
for config_file in "${config_files[@]}"; do
    # Extract the base name without extension for use in jobname
    config_name=$(basename "$config_file" .yaml)

    # Loop through each setting combination
    for setting in "${settings[@]}"; do
        # Evaluate the settings to create variables dynamically
        eval $setting


        # Dynamically create the job name
        jobname="${config_name}+bz=${per_gpu_batch_size}+lr=${learning_rate}+use_reconstruction_regularization=${use_reconstruction_regularization}+use_annealing=${use_annealing}+is_increasing=${is_increasing}"

        # Submit the job
        sbatch --job-name=$jobname --output="$output_root/$jobname/logs/slurm_%j.out" \
            exp_scripts/long_slurm.sh $config_name $per_gpu_batch_size $learning_rate $use_reconstruction_regularization $use_annealing $is_increasing $output_root
    done
done
