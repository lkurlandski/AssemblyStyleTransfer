#!/bin/bash -l

#SBATCH --job-name="dae"
#SBATCH --account=admalware
#SBATCH --partition=debug
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
# SBATCH --ntasks-per-node=4
# SBATCH --gres=gpu:a100:1
# SBATCH --mem=16G

export n_tasks_per_node=4
source ~/anaconda3/etc/profile.d/conda.sh
conda activate AssemblyStyleTransfer
python src/pretrain/dae.py \
--tok_n_files=10 \
--dat_n_files=10 \
--downsize=4 \
--output_dir="./output/models/dae" \
--overwrite_output_dir \
--do_train \
--do_eval \
--optim="adamw_torch" \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--save_total_limit=4 \
--num_train_epochs=100 \
--load_best_model_at_end \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--dataloader_num_workers=$n_tasks_per_node
