#!/usr/bin/env bash

# Usage:
#   bash run_eval.sh MODEL_PATH MODEL_NAME
#
# Example:
#   bash run_eval.sh \
#     checkpoints/156849_Fusion_DET_10_0.02881295018559293_16_0.0_1_exprt_235_seed_0_3-month-in-hospital-mortality_original_full_training_10/checkpoint_10 \
#     medfuse
#
# MODEL_PATH:
#   Path to the trained checkpoint you want to evaluate.
#
# MODEL_NAME:
#   Name used when saving inference / plot data.
#   Examples from previous runs:
#     medfuse
#     medfuse_mc_dropout
#     uninformative_prior
#     medcertain_1

MODEL_PATH="${1:?Error: please provide MODEL_PATH as the first argument}"
MODEL_NAME="${2:?Error: please provide MODEL_NAME as the second argument}"

MIMIC_TASK="${MIMIC_TASK:-3-month-in-hospital-mortality}"
SEED="${SEED:-0}"
JOBID="${JOBID:-001}"

python main.py \
  --config configs/nn-tdvi-pt-fusion-mimic-det.json \
  --jobid "$JOBID" \
  --model_for_final_eval "$MODEL_PATH" \
  --evaluate \
  --mimic_task "$MIMIC_TASK" \
  --seed "$SEED" \
  --inference_data_model_name "$MODEL_NAME" \
  --save_plot_data