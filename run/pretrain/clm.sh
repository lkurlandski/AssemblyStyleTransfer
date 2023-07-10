#!/bin/bash -l

#SBATCH --job-name="clm"
#SBATCH --account=admalware
#SBATCH --partition=debug
#SBATCH --output=./logs/%x_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G

export CUDA_VISIBLE_DEVICES=1
export n_tasks_per_node=1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate HMCST
python src/pretrain/clm.py \
--max_length=128 \
--vocab_size=4096 \
--tok_algorithm="BPE" \
--tok_use_saved=true \
--tok_overwrite=false \
--tok_batch_size=4096 \
--tok_n_files=5000 \
--dat_n_examples=2048 \
--dat_use_saved=true \
--dat_overwrite=false \
--dat_path="./output/pretrain" \
--num_proc=$n_tasks_per_node \
--scale=.75 \
--output_dir="./output/clm" \
--overwrite_output_dir=true \
--do_train=true \
--load_best_model_at_end=true \
--save_strategy="epoch" \
--evaluation_strategy="epoch" \
--num_train_epochs=100 \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=1024 \
--dataloader_num_workers=4 \
--hub_token="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"

