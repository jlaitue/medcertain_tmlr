#!/usr/bin/env bash

# Usage:
# You can also override variables like this:
#   MIMIC_TASK=3-month-in-hospital-mortality bash run_psvi_training.sh
#   PRETRAINED_PRIOR_PATH=/path/to/deterministic/medfuse/checkpoint bash run_psvi_training.sh
#
# IMPORTANT:
#   PRETRAINED_PRIOR_PATH should be the trained deterministic MedFuse model checkpoint.
#   This is the model used as the pretrained prior when --pretrained_prior is enabled.
#   Replace the path below with the checkpoint from your deterministic MedFuse run.

MIMIC_TASK="${1:-${MIMIC_TASK:-3-month-in-hospital-mortality}}"

PRETRAINED_PRIOR_PATH="${PRETRAINED_PRIOR_PATH:-GOLD_ckpts_summer_2026/156849_Fusion_DET_10_0.02881295018559293_16_0.0_1_exprt_235_seed_0_3-month-in-hospital-mortality_original_full_training_10/checkpoint_10}"

CONTEXT_POINTS_DATA_FILE="${CONTEXT_POINTS_DATA_FILE:-Context-III/in-hospital-mortality/context_set_3_cos_sim_type1_in-hospital-mortality_1_std_seed_0.npz}"

python main.py \
  --config configs/nn-tdvi-pt-fusion-mimic-psvi.json \
  --jobid 101 \
  --save_to_wandb \
  --full_training \
  --model_for_final_eval LAST \
  --learning_rate 0.0006465 \
  --batch_size 16 \
  --num_epochs 10 \
  --alpha 0 \
  --context_batch_size 16 \
  --context_points merged \
  --prior_var 1 \
  --prior_likelihood_scale 0.1 \
  --prior_likelihood_f_scale 0 \
  --prior_likelihood_cov_scale 0.01 \
  --prior_likelihood_cov_diag 0.5 \
  --mimic_task "$MIMIC_TASK" \
  --wandb_project medcertain-3-month \
  --seed 0 \
  --pretrained_prior \
  --context_points_data_file "$CONTEXT_POINTS_DATA_FILE" \
  --pretrained_prior_path "$PRETRAINED_PRIOR_PATH"