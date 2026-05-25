# MIMIC_TASK can be any of: in-hospital-mortality (48h), 3-month-in-hospital-mortality, 6-month-in-hospital-mortality, 1-year-in-hospital-mortality

MIMIC_TASK="${1:-${MIMIC_TASK:-3-month-in-hospital-mortality}}"

python main.py \
  --config configs/nn-tdvi-pt-fusion-mimic-det.json \
  --jobid 0 \
  --save_to_wandb \
  --full_training \
  --model_for_final_eval LAST \
  --batch_size 16 \
  --num_epochs 10 \
  --learning_rate 0.02881295018559293 \
  --alpha 0 \
  --reg_scale 1 \
  --mimic_task "$MIMIC_TASK" \
  --wandb_project uq-3-month \
  --seed 0