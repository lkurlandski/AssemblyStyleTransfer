#!/bin/bash -l

#SBATCH --job-name=train
#SBATCH --comment="train a seq2seq model"

#SBATCH --account=admalware
#SBATCH --partition=debug

#SBATCH --output=./slurm/%x/%j.out
#SBATCH --error=./slurm/%x/%j.err

#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G

export n_tasks_per_node=4
source ~/anaconda3/etc/profile.d/conda.sh
conda activate AssemblyStyleTransfer
torchrun --nproc-per-node 1 \
train.py \
--tr_unsupervised \
--mode="min" \
--tokenizer="WordLevel" \
--output_dir="I set this value manually in the code" \
--overwrite_output_dir \
--do_train \
--do_eval \
--optim="adamw_torch" \
--per_device_train_batch_size=512 \
--per_device_eval_batch_size=1024 \
--save_total_limit=3 \
--num_train_epochs=100 \
--fp16 \
--tf32=true \
--load_best_model_at_end \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--dataloader_num_workers=$n_tasks_per_node
