#!/bin/bash
echo starts
mount_from=/project/cosmos/haotiany/joint_training/
mount_to=/joint_training
container_path=/project/cosmos/haotiany/docker_images/imaginaire4_v9.2.2.sqsh

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 True True results_ablation titok_b512_4096_12+method=px+p_mean=0.5+anneal=True+causal=True px results_ablation  > results_ablation/logs/titok_b512_4096_12+method=px+p_mean=0.5+anneal=True+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 True False results_ablation titok_b512_4096_12+method=px+p_mean=0.5+anneal=True+causal=False px results_ablation  > results_ablation/logs/titok_b512_4096_12+method=px+p_mean=0.5+anneal=True+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 False True results_ablation titok_b512_4096_12+method=px+p_mean=0.5+anneal=False+causal=True px results_ablation  > results_ablation/logs/titok_b512_4096_12+method=px+p_mean=0.5+anneal=False+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 False False results_ablation titok_b512_4096_12+method=px+p_mean=0.5+anneal=False+causal=False px results_ablation  > results_ablation/logs/titok_b512_4096_12+method=px+p_mean=0.5+anneal=False+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 True True results_ablation titok_b512_4096_12+method=px+p_mean=0.25+anneal=True+causal=True px results_ablation  > results_ablation/logs/titok_b512_4096_12+method=px+p_mean=0.25+anneal=True+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 True False results_ablation titok_b512_4096_12+method=px+p_mean=0.25+anneal=True+causal=False px results_ablation  > results_ablation/logs/titok_b512_4096_12+method=px+p_mean=0.25+anneal=True+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 False True results_ablation titok_b512_4096_12+method=px+p_mean=0.25+anneal=False+causal=True px results_ablation  > results_ablation/logs/titok_b512_4096_12+method=px+p_mean=0.25+anneal=False+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 False False results_ablation titok_b512_4096_12+method=px+p_mean=0.25+anneal=False+causal=False px results_ablation  > results_ablation/logs/titok_b512_4096_12+method=px+p_mean=0.25+anneal=False+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 True True results_ablation titok_b512_4096_12+method=titok+p_mean=0.5+anneal=True+causal=True titok results_ablation  > results_ablation/logs/titok_b512_4096_12+method=titok+p_mean=0.5+anneal=True+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 True False results_ablation titok_b512_4096_12+method=titok+p_mean=0.5+anneal=True+causal=False titok results_ablation  > results_ablation/logs/titok_b512_4096_12+method=titok+p_mean=0.5+anneal=True+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 False True results_ablation titok_b512_4096_12+method=titok+p_mean=0.5+anneal=False+causal=True titok results_ablation  > results_ablation/logs/titok_b512_4096_12+method=titok+p_mean=0.5+anneal=False+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 False False results_ablation titok_b512_4096_12+method=titok+p_mean=0.5+anneal=False+causal=False titok results_ablation  > results_ablation/logs/titok_b512_4096_12+method=titok+p_mean=0.5+anneal=False+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 True True results_ablation titok_b512_4096_12+method=titok+p_mean=0.25+anneal=True+causal=True titok results_ablation  > results_ablation/logs/titok_b512_4096_12+method=titok+p_mean=0.25+anneal=True+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 True False results_ablation titok_b512_4096_12+method=titok+p_mean=0.25+anneal=True+causal=False titok results_ablation  > results_ablation/logs/titok_b512_4096_12+method=titok+p_mean=0.25+anneal=True+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 False True results_ablation titok_b512_4096_12+method=titok+p_mean=0.25+anneal=False+causal=True titok results_ablation  > results_ablation/logs/titok_b512_4096_12+method=titok+p_mean=0.25+anneal=False+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 False False results_ablation titok_b512_4096_12+method=titok+p_mean=0.25+anneal=False+causal=False titok results_ablation  > results_ablation/logs/titok_b512_4096_12+method=titok+p_mean=0.25+anneal=False+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 True True results_ablation titok_b512_4096_12+method=elastic+p_mean=0.5+anneal=True+causal=True elastic results_ablation  > results_ablation/logs/titok_b512_4096_12+method=elastic+p_mean=0.5+anneal=True+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 True False results_ablation titok_b512_4096_12+method=elastic+p_mean=0.5+anneal=True+causal=False elastic results_ablation  > results_ablation/logs/titok_b512_4096_12+method=elastic+p_mean=0.5+anneal=True+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 False True results_ablation titok_b512_4096_12+method=elastic+p_mean=0.5+anneal=False+causal=True elastic results_ablation  > results_ablation/logs/titok_b512_4096_12+method=elastic+p_mean=0.5+anneal=False+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.5 False False results_ablation titok_b512_4096_12+method=elastic+p_mean=0.5+anneal=False+causal=False elastic results_ablation  > results_ablation/logs/titok_b512_4096_12+method=elastic+p_mean=0.5+anneal=False+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 True True results_ablation titok_b512_4096_12+method=elastic+p_mean=0.25+anneal=True+causal=True elastic results_ablation  > results_ablation/logs/titok_b512_4096_12+method=elastic+p_mean=0.25+anneal=True+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 True False results_ablation titok_b512_4096_12+method=elastic+p_mean=0.25+anneal=True+causal=False elastic results_ablation  > results_ablation/logs/titok_b512_4096_12+method=elastic+p_mean=0.25+anneal=True+causal=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 False True results_ablation titok_b512_4096_12+method=elastic+p_mean=0.25+anneal=False+causal=True elastic results_ablation  > results_ablation/logs/titok_b512_4096_12+method=elastic+p_mean=0.25+anneal=False+causal=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group_causal.sh titok_b512_4096_12 32 2e-4 True True 0.25 False False results_ablation titok_b512_4096_12+method=elastic+p_mean=0.25+anneal=False+causal=False elastic results_ablation  > results_ablation/logs/titok_b512_4096_12+method=elastic+p_mean=0.25+anneal=False+causal=False.log 2>&1 &
sleep 0.5
wait
echo finished
