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
output_root="results_try_new_design"
# policy_network="mlp"

# Loop through each YAML file
for config_file in "${config_files[@]}"; do
    # Extract the base name without extension for use in jobname
    config_name=$(basename "$config_file" .yaml)

    # Loop through each setting combination
    # for setting in "${settings[@]}"; do
        # Evaluate the settings to create variables dynamically
        # eval $setting
    for use_ours in "True"; do
        for alpha_start in 0 0.1 2; do
            for policy_network in "mlp"; do
                for rate_weight in 0.1 0.5 1; do    
        # alpha_start=0.0
        # policy_network="mlp"
        # rate_weight=1.0

                    # Dynamically create the job name
                    # jobname="elastic+${config_name}"
                    jobname="gaussian+rate_weight=${rate_weight}+policy_network=${policy_network}+anneal_policy+alpha_start=${alpha_start}"

                    echo "jobname = $jobname"
                    # Submit the job
                    command="sbatch --job-name=$jobname --output='$output_root/$jobname/logs/slurm_%j.out' \
                        exp_scripts/long_slurm.sh $config_name $per_gpu_batch_size $learning_rate $use_reconstruction_regularization $use_ours $rate_weight $policy_network $alpha_start $output_root"
                    echo "$command"
                    eval "$command"
                # exit
                done
            done
        done
    done
done
