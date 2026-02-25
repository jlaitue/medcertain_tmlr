import argparse
import random

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fmnist')  # cifar10, cifar10-224, cifar100, fmnist, two-moons
    parser.add_argument('--prediction_type', type=str, default='classification')
    parser.add_argument('--mimic_task', type=str, default='phenotyping')
    parser.add_argument('--mimic_data_pairing', type=str, default='paired_ehr_cxr') # Any of -> radiology | paired_ehr_cxr | paired_ehr | partial_ehr | partial_ehr_cxr
    parser.add_argument("--full_training", action="store_true", default=False) # Train the model with all train+val sets for to obtain final model after finetuning
    parser.add_argument("--det_model_fine_tuning", action="store_true", default=False)
    parser.add_argument("--set_zero_gradients", action="store_true", default=False)
    parser.add_argument("--pretrained_prior", action="store_true", default=False)
    parser.add_argument("--pretrained_prior_path", type=str, default="")
    parser.add_argument("--inference_data_model_name", type=str, default="") # Defines the name of the model file to save inference data
    parser.add_argument("--batch_size", type=int, default=16) # CAREFUL WITH THIS BECAUSE OF DATALOADER
    parser.add_argument("--label_file_splits", type=str, default="original") # Any of: original | seed_test | same_test | medfuse_test
    parser.add_argument("--context_batch_size", type=int, default=16) # CAREFUL WITH THIS BECAUSE OF DATALOADER
    parser.add_argument("--training_dataset_size", type=int, default=0)
    parser.add_argument("--context_dataset_size", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--learning_rate_scale_logvar", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--exponent", type=float, default=1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument('--optimizer_name', type=str, default='adam')  # sgd, adam, adamw
    parser.add_argument('--model_name', type=str, default='ResNet18')  # ResNet9, ResNet18, ResNet18-Pretrained, ResNet50-Pretrained
    parser.add_argument('--method', type=str, default='fsmap')  # fsmap, psmap, fsvi, psvi
    parser.add_argument('--reg_type', type=str, default='function_prior')  # function_prior, function_norm, parameter_norm, feature_parameter_norm, function_kl, parameter_kl
    parser.add_argument('--forward_points', type=str, default='train')
    parser.add_argument('--reg_points', type=str, default='train')
    parser.add_argument("--context_points_data_file", type=str, default="")
    parser.add_argument('--context_points', type=str, default='train')
    parser.add_argument("--context_transform", action="store_true", default=False)
    parser.add_argument("--corrupted", action="store_true", default=False)
    parser.add_argument('--ood_points', type=str, default='svhn')
    parser.add_argument("--mc_samples_llk", type=int, default=1)
    parser.add_argument("--mc_samples_reg", type=int, default=1)
    parser.add_argument("--mc_samples_eval", type=int, default=1)
    parser.add_argument("--reg_scale", type=float, default=1)
    parser.add_argument("--reg_scale_gap", type=float, default=1)
    parser.add_argument("--gap_regularizer", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--prior_mean", type=float, default=0)
    parser.add_argument("--prior_var", type=float, default=0)
    parser.add_argument("--prior_likelihood_scale", type=float, default=1)
    parser.add_argument("--prior_likelihood_f_scale", type=float, default=1)
    parser.add_argument("--prior_likelihood_cov_scale", type=float, default=0)
    parser.add_argument("--prior_likelihood_cov_diag", type=float, default=0)
    parser.add_argument("--prior_likelihood_mean", type=float, default=0)
    parser.add_argument("--prior_likelihood_normalize_feature", action="store_true", default=False)
    parser.add_argument("--likelihood_scale", type=float, default=1)
    parser.add_argument("--output_var", action="store_true", default=False)
    parser.add_argument("--rho_sam", type=float, default=0)
    parser.add_argument("--prior_params_var", type=float, default=1)
    parser.add_argument("--init_logvar", type=float, default=-50)
    parser.add_argument("--init_final_layer_weights_logvar", type=float, default=-50)
    parser.add_argument("--init_final_layer_bias_logvar", type=float, default=-50)
    parser.add_argument("--prior_feature_logvar", type=float, default=-50)
    parser.add_argument("--prior_precision", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--linearize", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--model_for_final_eval", type=str, default="BEST")
    parser.add_argument("--final_eval", action="store_true", default=False)
    parser.add_argument("--eval_patient_groups", action="store_true", default=False)
    parser.add_argument("--build_context_set", action="store_true", default=False)
    parser.add_argument("--restore_checkpoint", action="store_true", default=False)
    parser.add_argument('--batch_stats_init_epochs', type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug_print", action="store_true", default=False)
    parser.add_argument("--debug_psd", action="store_true", default=False)
    parser.add_argument("--log_frequency", type=int, default=5)
    parser.add_argument("--save_model_on", type=str, default="AUROC") # Any of AUROC | AUROC_AUPRC | AUROC_AUPRC
    parser.add_argument("--save_to_wandb", action="store_true", default=False)
    parser.add_argument("--save_plot_data", action="store_true", default=False)
    parser.add_argument('--wandb_project', type=str, default='uq-medfuse')
    parser.add_argument('--wandb_account', type=str, default='default')
    parser.add_argument('--gpu_mem_frac', type=float, default=0)
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--config_id', type=int, default=0)
    parser.add_argument('--config_name', type=str, default='')
    parser.add_argument('--cwd', type=str, default='')
    parser.add_argument("--jobid", type=str, default=f"live_sess_{random.randint(0,1000)}")
    parser.add_argument("--confidence_intervals", type=bool, default=False)

    # Parse Fusion context dataset hyperparams
    # CXR context dataset
    parser.add_argument("--cxr_context_randcrop_size", type=int, default=28)
    parser.add_argument("--cxr_context_randhorizontalflip_prob", type=int, default=0.5)
    parser.add_argument("--cxr_context_randverticalflip_prob", type=int, default=0.5)
    parser.add_argument("--cxr_context_gaussianblur_kernel", type=int, default=3)
    parser.add_argument("--cxr_context_randsolarize_threshold", type=int, default=0.5)
    parser.add_argument("--cxr_context_randsolarize_prob", type=int, default=0.5)
    parser.add_argument("--cxr_context_randinvert_prob", type=int, default=0.5)
    parser.add_argument("--cxr_context_colorjitter_brightness", type=int, default=0.5)
    parser.add_argument("--cxr_context_colorjitter_contrast", type=int, default=0.5)

    # EHR context dataset
    parser.add_argument("--ehr_context_dropstart_max_percent", type=int, default=0.4)
    parser.add_argument("--ehr_context_gaussian_mean", type=int, default=1)
    parser.add_argument("--ehr_context_gaussian_std", type=int, default=0)
    parser.add_argument("--ehr_context_gaussian_max_cols", type=int, default=76)

    parser.add_argument("--ssm", action="store_true", default=False)
    parser.add_argument('--primary_type', type=str, default='')
    parser.add_argument('--secondary_type', type=str, default='')
    parser.add_argument('--tertiary_type', type=str, default='')

    return parser