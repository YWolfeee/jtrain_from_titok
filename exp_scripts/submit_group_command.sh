
# config_name="titok_b512_4096_12.yaml"
# per_gpu_batch_size=32
# learning_rate=2e-4
# method="ours"
# output_root="results_causal"
# wandb_projects="temp"
# init_weight="results_try_new_design/titok_b512_4096_12+elbo_mode=+nll_only=0.5+rate_weight=1+elbo_lower=0.5+elbo_upper=0.5/checkpoint-250000/unwrapped_model/pytorch_model.bin"


#### BASELINE
# python exp_scripts/group_job_dispatcher.py titok_b512_4096_12.yaml 32 2e-4 baseline results_final try_paper_runs


#### OURS

# python exp_scripts/group_job_dispatcher.py titok_b512_4096_12.yaml 32 2e-4 ours results_finetune try_finetune_runs exp_scripts/for_long_slurm_ours_b512_start_mean=0.5.sh checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin
# job_name="20N@tokenizer_training@ours_b512_start_mean=0.5+nodes=20@P_p1"
# call_file="exp_scripts/for_long_slurm_ours_b512_start_mean=0.5.sh"
# nodes=20


#### python exp_scripts/group_job_dispatcher.py titok_b512_4096_12.yaml 32 2e-4 ours results_causal try_results_causal exp_scripts/for_long_slurm_ours_b512_causal.sh
# job_name="4N@tokenizer_training@ours_b512_start_mean=0.5+causal+nodes=4@P_p1"
# call_file="exp_scripts/for_long_slurm_ours_b512_causal.sh"
# nodes=4

#### python exp_scripts/group_job_dispatcher.py titok_b512_4096_12 32 2e-4 all results_ablation results_ablation exp_scripts/for_long_slurm_all_b512_ablate.sh
# output_root="results_ablation"
# job_name="24N@tokenizer_training@all_b512_ablate+nodes=24@P_p1"
# call_file="exp_scripts/for_long_slurm_all_b512_ablate.sh"
# nodes=24


#### CONTINUOUS
#### python exp_scripts/group_job_dispatcher.py titok_b256_4096_12 128 2e-4 continuous results_continuous results_continuous exp_scripts/for_long_slurm_all_b256_continuous.sh
output_root="results_continuous"
nodes=16
job_name="${nodes}N@tokenizer_training@all_b256_continuous+nodes=${nodes}@P_p1"
call_file="exp_scripts/for_long_slurm_all_b256_continuous.sh"

command="sbatch --job-name=${job_name} --output=${output_root}/${job_name}/logs/slurm_%j.out --nodes=${nodes} exp_scripts/long_slurm_group.sh $call_file $output_root "
echo $command
eval $command
