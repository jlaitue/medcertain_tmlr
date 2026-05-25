# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "data"

# Path to the folder where the trained models are saved
CHECKPOINT_PATH = "checkpoints"

# Path to location of MIMIC datasets on server
MIMIC_CXR_PATH = "data/physionet.org/files/mimic-cxr-jpg/2.0.0/"
MIMIC_EHR_PATH = "data/mimic-iv-extracted/"
MIMIC_PATIENT_METADATA_FILE = "data/physionet.org/files/mimic-iv-1.0/core/patients.csv"

# Setting general epsilon/jitter values
jitter = eps = EPSILON = 1e-6

EPSILON = 1e-6

LABEL_TITLES = {
    0 : "Acute and unspecified renal failure",
    1 : "Acute cerebrovascular disease",
    2 : "Acute myocardial infarction",
    3 : "Cardiac dysrhythmias",
    4 : "Chronic kidney disease",
    5 : "Chronic obstructive pulmonary disease",
    6 : "Complications of surgical procedures",
    7 : "Conduction disorders",
    8 : "Congestive heart failure; nonhypertensive",
    9 : "Coronary atherosclerosis and other heart disease",
    10 : "Diabetes mellitus with complications",
    11 : "Diabetes mellitus without complication",
    12 : "Disorders of lipid metabolism",
    13 : "Essential hypertension",
    14 : "Fluid and electrolyte disorders",
    15 : "Gastrointestinal hemorrhage",
    16 : "Hypertension with complications",
    17 : "Other liver diseases",
    18 : "Other lower respiratory disease",
    19 : "Other upper respiratory disease",
    20 : "Pleurisy; pneumothorax",
    21 : "Pneumonia (except caused by tuberculosis or std)",
    22 : "Respiratory failure",
    23 : "Septicemia (except in labor)",
    24 : "Shock"
}

NUM_CLASSES = {
    "in-hospital-mortality": 1,
    "1-year-in-hospital-mortality": 1,
    "3-month-in-hospital-mortality": 1,
    "6-month-in-hospital-mortality": 1,
    "los_7_days": 1,
    "phenotyping": 25,
    "radiology": 25
    }

MODEL_HPARAMS = [
    "num_classes",
    "c_hidden",
    "num_blocks",
    "act_fn",
    "block_class",
    "inference_data_model_name"
]

OBJECTIVE_HPARAMS = [
    "method",
    "stochastic",
    "mc_dropout",
    "reg_type",
    "reg_scale",
    "reg_scale_gap",
    "upscale_mimic_loss",
    "gap_regularizer",
    "prior_mean",
    "prior_var",
    "prior_likelihood_scale",
    "prior_likelihood_f_scale",
    "prior_likelihood_cov_scale",
    "prior_likelihood_cov_diag",
    "prior_likelihood_mean",
    "prior_likelihood_normalize_feature",
    "likelihood_scale",
    "rho_sam",
    "context_points",
    "forward_points",
    "reg_points",
    "mc_samples_llk",
    "mc_samples_reg",
    "training_dataset_size",
    "batch_size",
    "init_logvar",
    "init_final_layer_weights_logvar",
    "init_final_layer_bias_logvar",
    "prior_feature_logvar",
    "pretrained_prior",
    "det_model_fine_tuning",
    ]

OTHER_HPARAM_KEYS = [
    "linearize",
    "output_var",
    "stochastic",
    "evaluate",
    "restore_checkpoint",
    "batch_stats_init_epochs",
    "dataset",
    "prediction_type",
    "mimic_task",
    "ood_points",
    "inference_data_model_name",
    "context_batch_size",
    "context_dataset_size",
    "num_epochs",
    "seed",
    "jobid",
    "mc_samples_eval",
    "config_name",
    "debug_print",
    "log_frequency",
    "final_eval",
    "eval_patient_groups",
    "save_model_on",
    "save_to_wandb",
    "wandb_project",
    "wandb_account",
    "confidence_intervals",
    "label_file_splits",
    "full_training",
    "model_for_final_eval",
    "pretrained_prior_path",
    "build_context_set",
    "set_zero_gradients",
    "save_plot_data",
]

SSM_HPARAMS = [       
    "ssm",
    "primary_type",
    "secondary_type",
    "tertiary_type",
]

OPTIMIZER_HPARAMS = [
    "lr",
    "b1",
    "b2",
    "learning_rate_scale_logvar",
    "momentum",
    "alpha",
    "exponent",
    "weight_decay",
]

def make_logger():
    init_zero = {
        "loss_train",
        "loss_test",
        "acc_train",
        "auroc_test",
        "auroc_selective_test",
        "auprc_test",
        "auprc_selective_test",
        "auroc_auprc_test",
        "nll_test",
        "nll_median_test",
        "nll_sum_test",
        "ece_test",
    }

    keys = [
        "epoch",
        "loss_train", "loss_val", "loss_test",
        "acc_train",

        # AUROC
        "auroc_test", "auroc_ci_test",
        "auroc_val", "auroc_ci_val",
        "auroc_val_best", "auroc_ci_val_best",
        "auroc_selective_test", "auroc_selective_val", "auroc_selective_val_best",

        # AUPRC
        "auprc_test", "auprc_ci_test",
        "auprc_val", "auprc_ci_val",
        "auprc_val_best", "auprc_ci_val_best",
        "auprc_selective_test", "auprc_selective_val",

        "auroc_auprc_val", "auroc_auprc_val_best", "auroc_auprc_test",

        # NLL
        "nll_test", "nll_val",
        "nll_median_test", "nll_median_val",
        "nll_sum_test", "nll_sum_val",

        # ECE
        "ece_test", "ece_val",
    ]

    return {key: [0] if key in init_zero else [] for key in keys}