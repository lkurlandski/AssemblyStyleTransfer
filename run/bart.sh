#!/bin/bash -l

#SBATCH --job-name=pretrain
#SBATCH --comment="Train bart :)"

#SBATCH --account=admalware
#SBATCH --partition=debug

#SBATCH --mail-type=ALL
#SBATCH --mail-user=lk3591@g.rit.edu

#SBATCH --output=./slurm/%x/%j.out

#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

export n_tasks_per_node=6
source ~/anaconda3/etc/profile.d/conda.sh
conda activate AssemblyStyleTransfer
#torchrun \
#--standalone \
#--nnodes=1 \
#--nproc-per-node 2 \
#--rdzv-endpoint=localhost:29501 \
python \
bart.py \
--downsize=4 \
--max_seq_length=128 \
--n_files=10 \
--preprocessing_num_workers=1 \
--do_train=true \
--output_dir="./output/models/bart" \
--overwrite_output_dir \
--dataloader_num_workers=$n_tasks_per_node \
--evaluation_strategy="epoch" \
--save_strategy="epoch" \
--load_best_model_at_end=true \
--save_total_limit=2 \
--num_train_epochs=100 \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8
# --gradient_accumulation_steps=1 \
# --eval_accumulation_steps=1

