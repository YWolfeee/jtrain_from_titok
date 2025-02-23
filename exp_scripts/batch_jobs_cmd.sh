#!/bin/bash

# List of YAML configuration files (manually specify here)
config_files=(
    "titok_s256_4096_12.yaml"
    "titok_s512_4096_12.yaml"
    # "titok_b128_4096_12.yaml"
    # "titok_b256_4096_12.yaml"
    # "titok_b512_4096_12.yaml"
    # "titok_b512_4096_12_no_crop.yaml"
    # "titok_l128_4096_12.yaml"
    # "titok_l256_4096_12.yaml"
    # "titok_l512_4096_12.yaml"
    # Add or remove YAML files here
)

# Parameters to iterate over for each YAML file
settings=(
    "elbo_upper=1.0 elbo_lower=0.0"
    # "elbo_upper=0.8 elbo_lower=0.0"
    "elbo_upper=0.5 elbo_lower=0.5"
    # "use_reconstruction_regularization=False use_annealing=False use_self_distilliation=False"
    # "use_reconstruction_regularization=True use_annealing=False use_self_distilliation=False"
    # "use_reconstruction_regularization=True use_annealing=False use_self_distilliation=True"
    # "use_reconstruction_regularization=True use_annealing=True use_self_distilliation=False"
    # "use_reconstruction_regularization=True use_annealing=True use_self_distilliation=True"
)

# Define other parameters
use_reconstruction_regularization=True
per_gpu_batch_size=64
learning_rate=4e-4
is_increasing=False
output_root="results_try_new_design"
# elbo_lower="mlp"
# rate_weight=1.0
use_ours="True"

# Loop through each YAML file
for config_file in "${config_files[@]}"; do
    # Extract the base name without extension for use in jobname
    config_name=$(basename "$config_file" .yaml)

    # Loop through each setting combination
    for setting in "${settings[@]}"; do
        # Evaluate the settings to create variables dynamically
        eval $setting
    # for elbo_mode in "" "0.4+0.6" "upto_px"; do
    for elbo_mode in ""; do
        # for elbo_upper in 1.0; do
        #     for elbo_lower in 0.2; do
                for rate_weight in 1; do    
        # elbo_upper=0.0
        # elbo_lower="mlp"
        # rate_weight=1.0

                    # Dynamically create the job name
                    # jobname="elastic+${config_name}"
                    jobname="${config_name}+elbo_mode=${elbo_mode}+nll_only=0.5+rate_weight=${rate_weight}+elbo_lower=${elbo_lower}+elbo_upper=${elbo_upper}"

                    echo "jobname = $jobname"
                    # Submit the job
                    command="sbatch --job-name=$jobname --output='$output_root/$jobname/logs/slurm_%j.out' \
                        exp_scripts/long_slurm.sh $config_name $per_gpu_batch_size $learning_rate $use_reconstruction_regularization $use_ours $rate_weight $elbo_lower $elbo_upper $output_root $elbo_mode"
                    echo "$command"
                    eval "$command"
                # exit
                # done
            done
        done
    done
done
