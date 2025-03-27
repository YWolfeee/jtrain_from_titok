#!/bin/bash

SLURM_NNODES=$1
WORLD_SIZE=$2
MASTER_ADDR=$3
MASTER_PORT=$4

cd /joint_training/jtrain_from_titok
pwd
source ~/.bashrc  # 如果需要的话

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NODEID=$SLURM_NODEID"

# ping -c $MASTER_ADDR:$MASTER_PORT
# nc -zv [host] [port]
# nc -zv $MASTER_ADDR $MASTER_PORT

# 或者 telnet [host] [port]
# telnet $MASTER_ADDR $MASTER_PORT
