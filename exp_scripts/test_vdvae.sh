#PBS -N zexp_test_vdvae
#PBS -S /bin/bash
#PBS -l select=1:ncpus=6:mem=45gb:ngpus=1:host=cvml07

nvidia-smi
cd ~/jtrain_from_titok
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate titok

python ~/jtrain_from_titok/test_vdvae.py
