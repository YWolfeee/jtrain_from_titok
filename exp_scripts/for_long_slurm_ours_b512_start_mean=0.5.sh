#!/bin/bash
echo starts
mount_from=/project/cosmos/haotiany/joint_training/
mount_to=/joint_training
container_path=/project/cosmos/haotiany/docker_images/imaginaire4_v9.2.2.sqsh

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.5 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.5+anneal=False+finetune=False px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.5+anneal=False+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.5 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.5+anneal=False+finetune=True px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.5+anneal=False+finetune=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.5 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.5+anneal=True+finetune=False anneal_to_px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.5+anneal=True+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.5 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.5+anneal=True+finetune=True anneal_to_px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.5+anneal=True+finetune=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.4375 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.4375+anneal=False+finetune=False px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.4375+anneal=False+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.4375 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.4375+anneal=False+finetune=True px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.4375+anneal=False+finetune=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.4375 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.4375+anneal=True+finetune=False anneal_to_px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.4375+anneal=True+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.4375 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.4375+anneal=True+finetune=True anneal_to_px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.4375+anneal=True+finetune=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.375 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.375+anneal=False+finetune=False px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.375+anneal=False+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.375 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.375+anneal=False+finetune=True px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.375+anneal=False+finetune=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.375 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.375+anneal=True+finetune=False anneal_to_px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.375+anneal=True+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.375 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.375+anneal=True+finetune=True anneal_to_px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.375+anneal=True+finetune=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.25 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.25+anneal=False+finetune=False px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.25+anneal=False+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.25 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.25+anneal=False+finetune=True px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.25+anneal=False+finetune=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.25 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.25+anneal=True+finetune=False anneal_to_px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.25+anneal=True+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.25 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.25+anneal=True+finetune=True anneal_to_px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.25+anneal=True+finetune=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.125 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.125+anneal=False+finetune=False px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.125+anneal=False+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.125 False 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.125+anneal=False+finetune=True px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.125+anneal=False+finetune=True.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.125 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.125+anneal=True+finetune=False anneal_to_px try_finetune_runs  >> logs/titok_b512_4096_12+method=ours+p_mean=0.125+anneal=True+finetune=False.log 2>&1 &
sleep 0.5

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem-per-gpu=72G --gpus=8 --exclusive --container-mounts=$mount_from:$mount_to --container-image=$container_path /bin/bash $mount_to/jtrain_from_titok/exp_scripts/main_newloss_group.sh titok_b512_4096_12 32 2e-4 True True 0.125 True 0.5 results_finetune titok_b512_4096_12+method=ours+p_mean=0.125+anneal=True+finetune=True anneal_to_px try_finetune_runs checkpoints/titok_b512_4096_12+titok+p_mean=0.5.bin >> logs/titok_b512_4096_12+method=ours+p_mean=0.125+anneal=True+finetune=True.log 2>&1 &
sleep 0.5
wait
echo finished
