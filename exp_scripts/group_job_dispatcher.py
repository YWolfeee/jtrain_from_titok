import os
import sys

mount_from = "/project/cosmos/haotiany/joint_training/"
mount_to = "/joint_training"
container_path = "/project/cosmos/haotiany/docker_images/imaginaire4_v9.2.2.sqsh"
main_file = "exp_scripts/main_newloss_group_causal.sh"

config_name = sys.argv[1]
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
    # assert method != "ours"
    init_weight = ""
# n_nodes = sys.argv[8]

print(f"config_name={config_name}, batch_size={batch_size}, lr={lr}, method={method}, output_root={output_root}, wandb_projects={wandb_projects}, init_weight={init_weight}")

os.makedirs(output_root, exist_ok=True)

# if method == "ours":
#     assert int(n_nodes) == 1
# elif method == "baselines":
#     assert int(n_nodes) == 10

idx = 0
f = open(file_path, "w")
f.write("#!/bin/bash\n")
f.write("echo starts\n")

f.write(f"mount_from={mount_from}\n")
f.write(f"mount_to={mount_to}\n")
f.write(f"container_path={container_path}\n")

if method == "ours":
    for p_mean in [0.5, 0.25]:
    # for p_mean in [0.5, 0.4375, 0.375, 0.25, 0.125]:
        for anneal in [False, True]:
            for finetune in [False]:
                start_mean = 0.5
                job_name = f"{config_name}+method={method}+p_mean={p_mean}+anneal={anneal}+finetune={finetune}"
                this_weight = init_weight if finetune else ""

                if anneal:
                    elbo_mode = "anneal_to_px"
                else:
                    elbo_mode = "px"

                f.write("\n")
                command = f"srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/{main_file} {config_name} {batch_size} {lr} True True {p_mean} {anneal} {start_mean} {output_root} {job_name} {elbo_mode} {wandb_projects} {this_weight} >> logs/{job_name}.log 2>&1 &"
                f.write(command + "\n")
                f.write("sleep 0.5\n")

                # os.system(command)
                idx += 1
elif method =="baseline":
    for elbo_mode in ['titok', 'elastic']:
        for p_mean in [0.5, 0.4375, 0.375, 0.25, 0.125]:
            start_mean = 0.5
            job_name = f"{config_name}+method={elbo_mode}+p_mean={p_mean}"
            this_weight = ""
            anneal = False

            f.write("\n")
            command = f"srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/{main_file} {config_name} {batch_size} {lr} True True {p_mean} {anneal} {start_mean} {output_root} {job_name} {elbo_mode} {wandb_projects} {this_weight} > logs/{job_name}.log 2>&1 &"
            f.write(command + "\n")



f.write("wait\n")
f.write("echo finished\n")
f.close()
# os.system(command)