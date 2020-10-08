#!/bin/bash
#SBATCH -J model_test
#SBATCH --gres=gpu:2
#SBATCH -p DGXq
#SBATCH -w node18

margin=(1.0 2.0)
dataset_name=WN18
bern=(False True)
epochs=120
batch_size=(512 1024 2048)
learning_rate=0.01
dim=64
lr_decay=1.8
norm=1

#python3 ../main/main.py --dim 64 --bs 512 --init_lr 0.01 --lr_decay 1.8 --bern False --margin 1.0 --norm 1








rm "slurm-${SLURM_JOB_ID}.out"
