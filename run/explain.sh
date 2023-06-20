#!/bin/bash -l

#SBATCH --job-name=explain
#SBATCH --comment="download, clean, and disassemble data prior to learning"

#SBATCH --account=admalware
#SBATCH --partition=tier3

#SBATCH --output=./slurm/%x/%j.out
#SBATCH --error=./slurm/%x/%j.err

#SBATCH --time=4-00:00:00               # Time limit
#SBATCH --nodes=1                       # How many nodes to run on
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem=16G
#SBATCH --gres=gpu:p4:1


source ~/anaconda3/etc/profile.d/conda.sh
conda activate AssemblyStyleTransfer
python explain.py

cd /home/lk3591/Documents/code/MalConv2
conda activate MalConv2
python explain.py \
--config_file=/home/lk3591/Documents/code/AssemblyStyleTransfer/explain.ini \
--run \
--analyze \
--no_ben

find ./output/attributions -type f -name "summary.*" -exec mv {} ./output \;
find ./output/attributions -type f -name "*.pt" -exec mv {} ./output/attributions \;

