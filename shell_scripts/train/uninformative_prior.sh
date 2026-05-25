#!/usr/bin/env bash

# You can also override variables like this:
#   MIMIC_TASK=3-month-in-hospital-mortality bash run_mfvi_uninformative_prior.sh
#   PRETRAINED_PRIOR_PATH=/path/to/deterministic/medfuse/checkpoint bash run_mfvi_uninformative_prior.sh
#
# Notes:
#   This script is for the MFVI model with an uninformative prior.
#   prior_var is set to 1000.
#
# IMPORTANT:
#   PRETRAINED_PRIOR_PATH should point to the trained deterministic MedFuse checkpoint.
#   Replace the path below with the checkpoint from your deterministic MedFuse run.

MIMIC_TASK="${1:-${MIMIC_TASK:-3-month-in-hospital-mortality}}"

PRETRAINED_PRIOR_PATH="${PRETRAINED_PRIOR_PATH:-checkpoints/156849_Fusion_DET_10_0.02881295018559293_16_0.0_1_exprt_235_seed_0_3-month-in-hospital-mortality_original_full_training_10/checkpoint_10}"

python main.py \
  --config configs/nn-tdvi-pt-fusion-mimic-mfvi.json \
  --jobid 1 \
  --pretrained_prior \
  --save_to_wandb \
  --model_for_final_eval LAST \
  --batch_size 16 \
  --num_epochs 15 \
  --learning_rate 0.06465 \
  --alpha 0 \
  --prior_var 1000 \
  --mimic_task "$MIMIC_TASK" \
  --wandb_project uninformative-3mo \
  --seed 0 \
  --pretrained_prior_path "$PRETRAINED_PRIOR_PATH"