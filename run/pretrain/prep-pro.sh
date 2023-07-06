#!/bin/bash -l

#SBATCH --job-name="prep-pro"
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

export n_tasks_per_node=8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate AssemblyStyleTransfer
python src/pretrain/prep.py \
--max_length=128 \
--vocab_size=4096 \
--tok_algorithm="BPE" \
--tok_use_saved=true \
--tok_overwrite=false \
--tok_batch_size=4096 \
--tok_n_files=5000 \
--dat_use_saved=false \
--dat_overwrite=true \
--dat_path="./output/pretrain" \
--num_proc=$n_tasks_per_node
