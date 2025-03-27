#!/bin/bash

#SBATCH --job-name=multi_titok
#SBATCH --account=dir_cosmos_base
#SBATCH --partition=pool0_datahall_a
#SBATCH --nodes=2               # <-- 多节点
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00

# 这里根据自己需要决定是不是要指定 output 和 error 文件
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# 通过SLURM自动获取 MASTER_ADDR, MASTER_PORT, WORLD_SIZE
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# 下面这种写法，是用 jobID 后4位 + 10000 做 port，也可以自己写固定端口
export MASTER_PORT=$((10000 + $SLURM_JOB_ID % 10000))
export WORLD_SIZE=$(( SLURM_NNODES * 8 ))   # 节点数 * 每节点的 GPU 数

source ~/.bashrc
echo $me
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

# -------------------------------
# 关键：用 srun 启动一条命令，在容器里跑你的脚本
# -------------------------------
srun --export=ALL -l \
     --container-image=$me/docker_images/imaginaire4_v9.2.2.sqsh \
     --container-mounts=$me/joint_training/:/joint_training \
     bash /joint_training/jtrain_from_titok/exp_scripts/short_slurm_multinode.sh $SLURM_NNODES $WORLD_SIZE $MASTER_ADDR $MASTER_PORT
    #  bash /joint_training/jtrain_from_titok/temp.sh 0.1

exit_status=$?
echo "exit status code $exit_status"
