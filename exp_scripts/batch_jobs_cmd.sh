#!/bin/bash

# List of YAML configuration files (manually specify here)
config_files=(
    # "titok_b128_4096_12.yaml"
    "titok_b256_4096_12.yaml"
    # "titok_l128_4096_12.yaml"
    # "titok_l256_4096_12.yaml"
    # Add or remove YAML files here
)

# Parameters to iterate over for each YAML file
# settings=(
    # "use_reconstruction_regularization=False use_annealing=False use_self_distilliation=False"
    # "use_reconstruction_regularization=True use_annealing=False use_self_distilliation=False"
    # "use_reconstruction_regularization=True use_annealing=False use_self_distilliation=True"
    # "use_reconstruction_regularization=True use_annealing=True use_self_distilliation=False"
    # "use_reconstruction_regularization=True use_annealing=True use_self_distilliation=True"
# )

# Define other parameters
use_reconstruction_regularization=True
per_gpu_batch_size=64
learning_rate=4e-4
is_increasing=False
output_root="results_try_distortion_rate_tradeoff_loss"

# Loop through each YAML file
for config_file in "${config_files[@]}"; do
    # Extract the base name without extension for use in jobname
    config_name=$(basename "$config_file" .yaml)

    # Loop through each setting combination
    # for setting in "${settings[@]}"; do
        # Evaluate the settings to create variables dynamically
        # eval $setting
    for use_policy_annealing in "True" "False"; do
        for rate_weight in 0.01 0.05 0.1 0.5; do
            for policy_network in "mlp" "transformer"; do


                # Dynamically create the job name
                jobname="${config_name}+lr=4e-4+use_ours=${use_reconstruction_regularization}+use_policy_annealing=${use_policy_annealing}+rate_weight=${rate_weight}+policy_network=${policy_network}"

                echo "jobname = $jobname"
                # Submit the job
                command="sbatch --job-name=$jobname --output='$output_root/$jobname/logs/slurm_%j.out' \
                    exp_scripts/long_slurm.sh $config_name $per_gpu_batch_size $learning_rate $use_reconstruction_regularization $use_policy_annealing $rate_weight $policy_network $output_root"
                echo "$command"
                eval "$command"
                # exit
            done
        done
    done
done
