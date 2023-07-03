#!/bin/bash -l

#SBATCH --job-name=prepare
#SBATCH --comment="download, clean, and disassemble data prior to learning"

#SBATCH --account=admalware
#SBATCH --partition=tier3

#SBATCH --output=./slurm/%x/%j.out
#SBATCH --error=./slurm/%x/%j.err

#SBATCH --time=1-00:00:00		# Time limit
#SBATCH --nodes=1			# How many nodes to run on
#SBATCH --ntasks=1			# How many tasks per node
#SBATCH --cpus-per-task=1		# Number of CPUs per task
#SBATCH --mem=16G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate AssemblyStyleTransfer
python prepare.py \
--n_sorel_files=25 \
--n_windows_files=25 \
--_32_bit 
