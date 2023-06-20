#!/bin/bash -l

#SBATCH --job-name=preprocess
#SBATCH --comment="train tokenizers and preprocess datasets for learning"

#SBATCH --account=admalware
#SBATCH --partition=tier3

#SBATCH --output=./slurm/%x/%j.out
#SBATCH --error=./slurm/%x/%j.err

#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate AssemblyStyleTransfer
# python preprocess.py --no_cache_tokenizer
python preprocess.py --pretrain --no_cache_dataset
# python preprocess.py --unsupervised --no_cache_dataset

