#!/bin/bash -l

#SBATCH --job-name="prep-tok"
#SBATCH --account=admalware
#SBATCH --partition=tier3
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

export n_tasks_per_node=1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate AssemblyStyleTransfer
python src/pretrain/prep.py \
--max_length=128 \
--vocab_size=4096 \
--tok_algorithm="BPE" \
--tok_use_saved="false" \
--tok_overwrite="true" \
--tok_batch_size=4096 \
--tok_n_files=5000 \
--dat_n_examples=0 \
--dat_n_files=0 \
--num_proc=$n_tasks_per_node
