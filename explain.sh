#!/bin/bash -l

#SBATCH --job-name=prepare
#SBATCH --comment="download, clean, and disassemble data prior to learning"

#SBATCH --account=admalware
#SBATCH --partition=debug

#SBATCH --output=./slurm/%x/%j.out
#SBATCH --error=./slurm/%x/%j.err

#SBATCH --time=1-00:00:00               # Time limit
#SBATCH --nodes=1                       # How many nodes to run on
#SBATCH --ntasks=1                      # How many tasks per node
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH 

exit

cd /home/lk3591/Documents/code/MalConv2
/home/lk3591/anaconda3/envs/MalConv2/bin/python explain.py --config_file=/home/lk3591/Documents/code/HMCST/explain.ini --run --analyze --no_ben
