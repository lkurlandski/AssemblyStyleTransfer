#!/bin/bash -l

#SBATCH --job-name=bart_prep
#SBATCH --comment="Data processing for BART stuff"

#SBATCH --account=admalware
#SBATCH --partition=tier3

#SBATCH --output=./slurm/%x/%j.out
#SBATCH --error=./slurm/%x/%j.err

#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

export n_tasks_per_node=16
source ~/anaconda3/etc/profile.d/conda.sh
conda activate AssemblyStyleTransfer
python \
bart.py \
--overwrite=1 \
--output_dir="./output/models/bart" \
--max_seq_length=256 \
--downsize=4 \
--preprocessing_num_workers=$n_tasks_per_node \
--evaluation_strategy="epoch" \
--load_best_model_at_end=False \
--save_strategy="epoch" 
