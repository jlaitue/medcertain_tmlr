#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100
#SBATCH --time=1-05:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=200GB
#SBATCH --array=1-5
# Output and error files
#SBATCH -e slurm_logs/fusion_det_runs/errlogs/job.%J.err
#SBATCH -o slurm_logs/fusion_det_runs/outlogs/job.%J.out
# Email notifications
#SBATCH --mail-type=ALL
# Resource requirements commands

# Activating conda
eval "$(conda shell.bash hook)"
conda activate uq-wq

export XLA_PYTHON_CLIENT_MEM_FRACTION=".80"; \
export WANDB_API_KEY=; \
srun $(head -n $SLURM_ARRAY_TASK_ID job_files/gap/det.txt | tail -n 1)