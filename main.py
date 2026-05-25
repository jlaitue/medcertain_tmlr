from utils.config_setup import setup_script
args, rng_key = setup_script()
from utils.load_data import get_mimic_data
from utils.load_medcertain import setup_medcertain
from utils.constants import OTHER_HPARAM_KEYS, OBJECTIVE_HPARAMS, SSM_HPARAMS, OPTIMIZER_HPARAMS, MODEL_HPARAMS
import wandb
from copy import copy
#-----------------------------------------------------------------------------

train_loader, val_loader, test_loader, context_loader = get_mimic_data(args)

# Setup: model, trainer and optimizer
medcertain, kwargs = setup_medcertain(
    model_name=args["model_name"],
    model_class=args["model_class"],
    optimizer_name=args["optimizer_name"],

    model_hparams = {key: args[key] for key in MODEL_HPARAMS},
    optimizer_hparams = {key: args[key] for key in OPTIMIZER_HPARAMS},
    objective_hparams = {key: args[key] for key in OBJECTIVE_HPARAMS},
    ssm_hparams = {key: args[key] for key in SSM_HPARAMS},
    other_hparams = {key: args[key] for key in OTHER_HPARAM_KEYS},

    rng_key=rng_key,

    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    context_loader=context_loader,
    general_args=args
    )

if medcertain.save_to_wandb:
    wandb.config = copy(kwargs)
    wand_run = wandb.init(
        project=medcertain.wandb_project,
        name=medcertain.run_name,
        entity=medcertain.wandb_account,
        config=wandb.config,
    )
    
# Train model from scratch
if not medcertain.evaluate: 
    print("Starting model training...") 
    medcertain.train_model(train_loader, context_loader, val_loader, rng_key, num_epochs=medcertain.num_epochs)

# Load trained model for evaluation OR context set creation OR saving preds/targets
best_model_path = medcertain.get_best_model_path()
print(f"\nLoading {medcertain.model_for_final_eval} model for evaluation...")
print(f"\n  {best_model_path}")
# You can replace the path directly here or with the --model_for_final_eval arguments flag
# best_model_path = /path_to_best_ckpt

medcertain.load_best_model_ckpt(best_model_path, rng_key)

# Context set creation
if medcertain.build_context_set:
    print(f"\nExtracting TRAIN data for CONTEXT set construction...")
    medcertain.eval_model(rng_key, medcertain.n_batches_eval_final, final_eval=False, build_context_set=True, save_plot_data=False)

# Saving TEST preds/targets, no computation of metrics to make this faster.
elif medcertain.save_plot_data:
    print(f"\nSaving preds and targets from loaded model on TEST set...")
    medcertain.eval_model(rng_key, medcertain.n_batches_eval_final, final_eval=True, build_context_set=False, save_plot_data=True)

else:
    medcertain.eval_model(rng_key, medcertain.n_batches_eval_final, final_eval=True, build_context_set=False, save_plot_data=False)
    log = medcertain.logger
    print(
        f"\nFINAL TEST SET SCORES:"
        f"AUROC {log['auroc_test'][-1]*100:.3f} | "
        f"AUPRC {log['auprc_test'][-1]*100:.3f} | "
        f"SEL AUROC {log['auroc_selective_test'][-1]*100:.3f} | "
        f"SEL AUPRC {log['auprc_selective_test'][-1]*100:.3f} | "
    )
    print()
    if medcertain.save_to_wandb:
        medcertain.final_log_wandb()

print(f"\nFinished evaluation on best model checkpoint\n {best_model_path}")
print("------------------------------------------------------------\n\n")

print(f"\nExperiment done.\n")