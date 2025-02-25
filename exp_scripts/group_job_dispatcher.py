import os
import sys

config_path = sys.argv[1]
batch_size = sys.argv[2]
lr = sys.argv[3]
method = sys.argv[4]
output_root = sys.argv[5]
wandb_projects = sys.argv[6]
file_path=sys.argv[7]
# file_path="exp_scripts/for_long_slurm_ours_b512_p_mean=0.5.sh"
try:
    init_weight = sys.argv[8]
except:
    assert method != "ours"
    init_weight = ""
# n_nodes = sys.argv[8]

print(f"config_path={config_path}, batch_size={batch_size}, lr={lr}, method={method}, output_root={output_root}, wandb_projects={wandb_projects}, init_weight={init_weight}")

os.makedirs(output_root, exist_ok=True)

# if method == "ours":
#     assert int(n_nodes) == 1
# elif method == "baselines":
#     assert int(n_nodes) == 10

idx = 0
f = open(file_path, "w")
f.write("#!/bin/bash\n")
f.write("echo starts\n")

if method == "ours":
    for p_mean in [0.5]:
    # for p_mean in [0.5, 0.4375, 0.375, 0.25, 0.125]:
        for anneal in [False, True]:
            for finetune in [False, True]:
                start_mean = 0.5
                job_name = f"{config_path}+method={method}+p_mean={p_mean}+anneal={anneal}+finetune={finetune}"
                config_name = config_path.split(".")[0]
                this_weight = init_weight if finetune else ""

                if anneal:
                    elbo_mode = "anneal_to_px"
                else:
                    elbo_mode = "px"

                f.write("\n")
                command = f"srun --nodes=1 --ntasks=1 --gpus=8 --exclusive --container-mounts=/lustre/fsw/portfolios/dir/users/haotiany/joint_training/:/joint_training --container-image=./docker_images/imaginaire4_v9.2.2.sqsh /bin/bash /joint_training/jtrain_from_titok/exp_scripts/main_newloss.sh {config_path} {batch_size} {lr} True True {p_mean} {anneal} {start_mean} {output_root} {job_name} {elbo_mode} {wandb_projects} {this_weight} > logs/{job_name}.log 2>&1 &"
                f.write(command + "\n")
                # os.system(command)
                idx += 1
elif method =="baseline":
    for elbo_mode in ['titok', 'elastic']:
        for p_mean in [0.5, 0.4375, 0.375, 0.25, 0.125]:
            start_mean = 0.5
            job_name = f"{config_path}+method={elbo_mode}+p_mean={p_mean}"
            config_name = config_path.split(".")[0]
            this_weight = ""
            anneal = False

            f.write("\n")
            command = f"srun --nodes=1 --ntasks=1 --gpus=8 --exclusive --container-mounts=/lustre/fsw/portfolios/dir/users/haotiany/joint_training/:/joint_training --container-image=./docker_images/imaginaire4_v9.2.2.sqsh /bin/bash /joint_training/jtrain_from_titok/exp_scripts/main_newloss.sh {config_path} {batch_size} {lr} True True {p_mean} {anneal} {start_mean} {output_root} {job_name} {elbo_mode} {wandb_projects} {this_weight} > logs/{job_name}.log 2>&1 &"
            f.write(command + "\n")



f.write("wait\n")
f.write("echo finished\n")
f.close()
# os.system(command)