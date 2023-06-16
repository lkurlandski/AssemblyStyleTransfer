#!/bin/bash -l

#SBATCH --job-name=pretrain
#SBATCH --comment="indpendently train an encoder and a decoder"

#SBATCH --account=admalware
#SBATCH --partition=debug

#SBATCH --output=./slurm/%x/%j.out
#SBATCH --error=./slurm/%x/%j.err

#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate HMCST
torchrun --nproc-per-node 2 \
train.py \
--unsupervised \
--model="WordLevel"
