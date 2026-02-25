## Standard libraries
import os
import numpy as np
from PIL import Image
import pickle
from typing import Any
from collections import defaultdict
import wandb
import time
import tree
import random as random_py
import functools
from functools import partial
from copy import copy
from typing import (Any, Callable, Iterable, Optional, Tuple, Union, Dict)
import warnings
import h5py
import argparse
from tqdm.auto import tqdm
import json
from pprint import pprint
import re
import glob
from datetime import datetime
import math
import pandas as pd 
import pdb
## Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
# %matplotlib inline
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
rcParams['lines.linewidth'] = 2.0
# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman'] # need to have latex installed for this to work
rcParams['text.usetex'] = True
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
import seaborn as sns
sns.reset_orig()

## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import jit
from jax.config import config
# config.update("jax_debug_nans", True)
# config.update('jax_platform_name', 'cpu')

## Flax
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze, unfreeze
from flax.linen.initializers import lecun_normal

## JAX addons
import optax
import distrax
import neural_tangents as nt
import flaxmodels as fm
from flaxmodels.resnet import ops
from flaxmodels import utils

## Tensorflow
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
tfd = tfp.distributions

## PyTorch
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, MNIST, KMNIST, ImageNet
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn import datasets as sklearn_datasets

from pathlib import Path
from timm.data import create_dataset
from torch.utils.data import Dataset, random_split

## MedFuse code
from MedFuse.models import lstm_jax, fusion_jax
from MedFuse.datasets import cxr_dataset, ehr_dataset, fusion_dataset
from MedFuse.datasets import context_dataset as context_dataset_class
from MedFuse.ehr_utils.preprocessing import Discretizer, Normalizer
import MedFuse.confidence_intervals.ci_processing as medfuse_ci
import MedFuse.confidence_intervals.ci_processing_average as medfuse_ci_2

## Convert from CxHxW to HxWxC for Flax.
chw2hwc_fn = lambda img: img.permute(1, 2, 0)

#-----------------------------------------------------------------------------
from arguments import args_parser
from functions import (
    calibration, categorical_nll_with_softmax, gaussian_nll,
    mimic_ce_loss_sigmoid,
    selective_prediction_mimic,
    merge_params, split_params, numpy_collate,
    dual_vector
    )
from base_datasets import (
    CustomDataset,
    # MIMIC EHR
    get_discretizer, get_normalizer, load_MIMIC_EHR,
    # MIMIC CXR
    load_MIMIC_CXR
)
from base_architectures import (
    CNN, MLP_Toy, MLP_OfflineRL, ResNetBlock,
    PreActResNetBlock, ResNetMod, BasicBlock,
    Bottleneck, ResNet, ResNet18, ResNet34,
    ResNet50, ResNet101, ResNet152    
)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# PARSER
parser = args_parser()
args = parser.parse_args()
args_dict = vars(args)

config_file = args.config
config_id = args.config_id
config_name = args.config_name
cwd = os.getcwd()

if config_file != '':
    with open(config_file, 'r') as f:
        config_json = json.load(f)

    configurations = config_json['configurations']
    if config_name == '':
        name = configurations[config_id]['name']
    else:
        name = config_name
    id = configurations[config_id]['id']
    # cwd = configurations[config_id]['cwd']
    cwd = os.getcwd() + "/laboratory/"
    parser_args_list = configurations[config_id]['args']
    env_args = configurations[config_id]['env']

    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
        
    parser_args = {}

    for i in range(len(parser_args_list)):
        if parser_args_list[i].startswith('--'):
            key = parser_args_list[i][2:]
            value = parser_args_list[i+1]
            parser_args[key] = value

    print(f"\nCONFIG NAME: {name}")
    print(f"\nWORKING DIRECTORY: {cwd}")
    print(f"\nCONFIG ID: {id}")
    print(f"\nENVIRONMENT ARGS:\n\n{env_args}")

    for key in parser_args:
        args_dict[key] = parser_args[key]

    for key in parser_args:
        if isinstance(parser_args[key], int):
            args_dict[key] = int(parser_args[key])
        elif isinstance(parser_args[key], str) and parser_args[key].isnumeric():
            args_dict[key] = int(parser_args[key])
        elif isinstance(parser_args[key], str) and is_float(parser_args[key]):
            args_dict[key] = float(parser_args[key])
        elif parser_args[key] == 'True' or parser_args[key] == 'False':
            args_dict[key] = True if parser_args[key] == 'True' else False

    for key in env_args:
        os.environ[key] = env_args[key]

# Creating variables used throughout the script
dataset = args_dict["dataset"]
prediction_type = args_dict["prediction_type"]
mimic_task = args_dict["mimic_task"]
mimic_data_pairing = args_dict["mimic_data_pairing"]
full_training = args_dict["full_training"]
label_file_splits = args_dict["label_file_splits"]
batch_size = args_dict["batch_size"]
context_batch_size = args_dict["context_batch_size"]
training_dataset_size = args_dict["training_dataset_size"]
context_dataset_size = args_dict["context_dataset_size"]
num_epochs = args_dict["num_epochs"]
set_zero_gradients = args_dict["set_zero_gradients"]
inference_data_model_name = args_dict["inference_data_model_name"]
learning_rate = args_dict["learning_rate"]
learning_rate_scale_logvar = args_dict["learning_rate_scale_logvar"]
alpha = args_dict["alpha"]
exponent = args_dict["exponent"]
momentum = args_dict["momentum"]
optimizer_name = args_dict["optimizer_name"]
model_name = args_dict["model_name"]
method = args_dict["method"]
reg_type = args_dict["reg_type"]
gap_regularizer = args_dict["gap_regularizer"]
weight_decay = args_dict["weight_decay"]
context_points = args_dict["context_points"]
context_points_data_file = args_dict["context_points_data_file"]
forward_points = args_dict["forward_points"]
reg_points = args_dict["reg_points"]
context_transform = args_dict["context_transform"]
corrupted = args_dict["corrupted"]
ood_points = args_dict["ood_points"]
mc_samples_llk = args_dict["mc_samples_llk"]
mc_samples_reg = args_dict["mc_samples_reg"]
mc_samples_eval = args_dict["mc_samples_eval"]
reg_scale = args_dict["reg_scale"]
reg_scale_gap = args_dict["reg_scale_gap"]
prior_mean = args_dict["prior_mean"]
prior_var = args_dict["prior_var"]
prior_likelihood_scale = args_dict["prior_likelihood_scale"]
prior_likelihood_f_scale = args_dict["prior_likelihood_f_scale"]
prior_likelihood_cov_scale = args_dict["prior_likelihood_cov_scale"]
prior_likelihood_cov_diag = args_dict["prior_likelihood_cov_diag"]
prior_likelihood_mean = args_dict["prior_likelihood_mean"]
prior_likelihood_normalize_feature = args_dict["prior_likelihood_normalize_feature"]
likelihood_scale = args_dict["likelihood_scale"]
output_var = args_dict["output_var"]
rho_sam = args_dict["rho_sam"]
prior_params_var = args_dict["prior_params_var"]
init_logvar = args_dict["init_logvar"]
init_final_layer_weights_logvar = args_dict["init_final_layer_weights_logvar"]
init_final_layer_bias_logvar = args_dict["init_final_layer_bias_logvar"]
prior_feature_logvar = args_dict["prior_feature_logvar"]
prior_precision = args_dict["prior_precision"]
pretrained_prior = args_dict["pretrained_prior"]
det_model_fine_tuning = args_dict["det_model_fine_tuning"]
linearize = args_dict["linearize"]
seed = args_dict["seed"]
evaluate = args_dict["evaluate"]
pretrained_prior_path = args_dict["pretrained_prior_path"]
build_context_set = args_dict["build_context_set"]
model_for_final_eval = args_dict["model_for_final_eval"]
final_eval = args_dict["final_eval"]
eval_patient_groups = args_dict["eval_patient_groups"]
restore_checkpoint = args_dict["restore_checkpoint"]
batch_stats_init_epochs = args_dict["batch_stats_init_epochs"]
debug = args_dict["debug"]
debug_print = args_dict["debug_print"]
debug_psd = args_dict["debug_psd"]
log_frequency = args_dict["log_frequency"]
save_model_on = args_dict["save_model_on"]
save_to_wandb = args_dict["save_to_wandb"]
save_plot_data = args_dict["save_plot_data"]
wandb_project = args_dict["wandb_project"]
wandb_account = args_dict["wandb_account"]
gpu_mem_frac = args_dict["gpu_mem_frac"]
jobid = args_dict["jobid"]
confidence_intervals = args_dict["confidence_intervals"]
ssm = args_dict["ssm"]
primary_type = args_dict["primary_type"]
secondary_type = args_dict["secondary_type"]
tertiary_type = args_dict["tertiary_type"]

# Context dataset hyperparameters for CXR and EHR

# CXR context dataset hyperparams
cxr_context_hypers = {
    "randcrop_size": args_dict["cxr_context_randcrop_size"],
    "randhorizontalflip_prob": args_dict["cxr_context_randhorizontalflip_prob"],
    "randverticalflip_prob": args_dict["cxr_context_randverticalflip_prob"],
    "gaussianblur_kernel": args_dict["cxr_context_gaussianblur_kernel"],
    "randsolarize_threshold": args_dict["cxr_context_randsolarize_threshold"],
    "randsolarize_prob": args_dict["cxr_context_randsolarize_prob"],
    "randinvert_prob": args_dict["cxr_context_randinvert_prob"],
    "colorjitter_brightness": args_dict["cxr_context_colorjitter_brightness"],
    "colorjitter_contrast": args_dict["cxr_context_colorjitter_contrast"],
}

# EHR context dataset hyperparams
ehr_context_hypers = {
    "dropstart_max_percent": args_dict["ehr_context_dropstart_max_percent"],
    "gaussian_mean": args_dict["ehr_context_gaussian_mean"],
    "gaussian_std": args_dict["ehr_context_gaussian_std"],
    "gaussian_max_cols": args_dict["ehr_context_gaussian_max_cols"]
}

if cwd == "":
    raise Exception("Working directory has not been explicitly set.")
    
print(f"\PARSER ARGS:\n\n{args_dict}")
os.chdir(cwd)
# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "data"

# Path to the folder where the trained models are saved
CHECKPOINT_PATH = "checkpoints"

# Path to location of MIMIC datasets on server
MIMIC_CXR_PATH = "laboratory/data/physionet.org/files/mimic-cxr-jpg/2.0.0/" # MIMIC CXR dataset location
MIMIC_EHR_PATH = "laboratory/data/mimic-iv-extracted/" # MIMIC EHR dataset location
MIMIC_PATIENT_METADATA_FILE = "laboratory/data/physionet.org/files/mimic-iv-1.0/core/patients.csv"

# Setting general epsilon/jitter values
jitter = eps = 1e-6

# Seeding for random operations
print(f"\nSEED: {seed}")
main_rng = random.PRNGKey(seed)
rng_key = main_rng
os.environ["PYTHONHASHSEED"] = str(seed)
random_py.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.random.manual_seed(seed)

print(f"\nCURRENT COMPUTE DEVICE: {jax.devices()[0]}\n")

if debug: 
    config.update("jax_disable_jit", True)
# END OF GENERAL SETUP FOR SCRIPT
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# MIMIC DATASETS LOADING
if dataset == "mimic-cxr" and mimic_task == "radiology":
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("Creating MIMIC-CXR dataset for RADIOLOGY task...")

    # We need to set this, since we use the same dataloaders as the other datasets
    batch_size_test = batch_size
    labels_file = "mimic-cxr-pheno.csv"
    # labels_file = "mimic-cxr-2.0.0-chexpert.csv"
    num_classes = 25 if labels_file == "mimic-cxr-pheno.csv" else 14

    mimic_cxr_args = {
        "resize":256,
        "crop": 224, 
        "cxr_data_dir" : MIMIC_CXR_PATH,
        "split_file" : "mimic-cxr-ehr-split.csv",
        "metadata_file": "mimic-cxr-2.0.0-metadata.csv",
        "labels_file": labels_file,
        "labels_set": "radiology",
        # "seed": seed
        }

    DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    DATA_STD = np.array([0.247, 0.243, 0.261])

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img
    
    # ------------------------------------------------------------------------
    # MAIN CXR DATASET CREATION
    print("\nCreating MIMIC CXR datasets...")
    train_dataset, validation_dataset, test_dataset, training_dataset_size, val_dataset_size, test_dataset_size, paths = load_MIMIC_CXR(mimic_cxr_args)
    if full_training:
        train_dataset = data.ConcatDataset([train_dataset, validation_dataset])
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # CONTEXT DATASET CREATION
    context_transform_list = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(3,3)),
        transforms.RandomSolarize(threshold=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Resize(224), # We need 224x224 for MIMIC-CXR
    ]
    context_transform_list.append(image_to_numpy)
    context_transform = transforms.Compose(context_transform_list)

    if context_points == "train":
        context_dataset = cxr_dataset.MIMICCXR(paths, mimic_cxr_args, split="train", transform=context_transform)
    elif context_points == "svhn":
        context_dataset = SVHN(root=f"./{DATASET_PATH}/SVHN", split="train", transform=context_transform, download=False)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root=f"./{DATASET_PATH}/ImageNet", train=True, transform=context_transform, download=False)
    else:
        ValueError("Unknown context dataset")

    full_context_dataset_size = context_dataset_size = len(context_dataset)
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, full_context_dataset_size - context_dataset_size], generator=torch.Generator().manual_seed(seed))
    context_set = CustomDataset(context_set, training_dataset_size)
    context_dataset_size = len(context_set)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # OOD DATASET CREATION
    ood_transform_list = [transforms.Resize(224)]
    ood_transform_list.append(image_to_numpy)
    ood_transform = transforms.Compose(ood_transform_list)

    if ood_points == "svhn":
        ood_dataset = SVHN(root=f"./{DATASET_PATH}/SVHN", split="test", transform=ood_transform,  download=True)
    elif ood_points == "cifar100":
        ood_dataset = CIFAR100(root=f"./{DATASET_PATH}/CIFAR100", train=False, transform=ood_transform,  download=False)
    else:
        ValueError("Unknown OOD dataset")

    indices = torch.arange(1000)
    ood_dataset = data.Subset(ood_dataset, indices)
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate)
    # ------------------------------------------------------------------------

    print(f" Datasets lengths\n Training: {training_dataset_size} | Validation: {val_dataset_size} | Testing: {test_dataset_size} | Context: {context_dataset_size}")
    print("CXR datasets done.")
    print()

elif dataset == "mimic-fusion" or ((dataset == "mimic-cxr" or dataset == "mimic-ehr") and (mimic_task == "in-hospital-mortality" or mimic_task == "phenotyping")):
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("Creating FUSION-CXR-EHR datasets...")

    if mimic_task == "phenotyping":
        num_classes = 25
        labels_set = "phenotyping"
        # labels_file = "mimic-cxr-pheno.csv"
        labels_file = "mimic-cxr-2.0.0-chexpert.csv"
    elif mimic_task == "in-hospital-mortality":
        num_classes = 1
        labels_set = "mortality" # This is not used anywhere either in this code or original Medfuse
        labels_file = "mimic-cxr-2.0.0-chexpert.csv"
    else:
        ValueError("Unknown MIMIC task")
    
    mimic_fusion_args = {
        # Data folders
        "cxr_data_dir" : MIMIC_CXR_PATH,
        "ehr_data_dir" : MIMIC_EHR_PATH,
        # EHR args
        "task": mimic_task,
        "timestep": 1.0,
        "normalizer_state": None,
        # CXR args
        "resize":256, "crop": 224,
        "metadata_file": "mimic-cxr-2.0.0-metadata.csv",
        "split_file" : "mimic-cxr-ehr-split.csv",
        "labels_file": labels_file,
        # Fusion args
        "num_classes": num_classes,
        "data_pairs": "paired_ehr_cxr",
        "data_ratio": 1,
        "labels_set": labels_set,
        "label_file_splits": label_file_splits,
        "seed": seed,
        "full_training": full_training
        }

    print(f"\nARGS: {mimic_fusion_args}\n")

    # ------------------------------------------------------------------------
    # Step 1) EHR DATASET CREATION
    print("1. Creation of EHR datasets...")

    ehr_train_ds, ehr_val_ds, ehr_test_ds, ehr_training_dataset_size, ehr_val_dataset_size, ehr_test_dataset_size = load_MIMIC_EHR(mimic_fusion_args)

    print(f" Datasets lengths\n Training: {ehr_training_dataset_size} | Validation: {ehr_val_dataset_size} | Testing: {ehr_test_dataset_size}")
    print(" EHR datasets done.")
    print()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Step 2) CXR DATASET CREATION
    print("2. Creation of CXR datasets...")

    cxr_train_ds, cxr_val_ds, cxr_test_ds, cxr_training_dataset_size, cxr_val_dataset_size, cxr_test_dataset_size, cxr_paths = load_MIMIC_CXR(mimic_fusion_args)
    
    print(f" Datasets lengths\n Training: {cxr_training_dataset_size} | Validation: {cxr_val_dataset_size} | Testing: {cxr_test_dataset_size}")
    print(" CXR datasets done.")
    print()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Step 3) FUSION DATASET CREATION
    print("3. Creation of FUSION datasets...")

    train_dataset, validation_dataset, test_dataset = fusion_dataset.load_cxr_ehr(mimic_fusion_args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)

    if full_training:
        train_dataset = data.ConcatDataset([train_dataset, validation_dataset])

    training_dataset_size = len(train_dataset)
    val_dataset_size = len(validation_dataset)
    test_dataset_size = len(test_dataset)

    print(f" Datasets lengths\n Training: {training_dataset_size} | Validation: {val_dataset_size} | Testing: {test_dataset_size}")
    print(" FUSION datasets done.")
    print()
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Step 4) FUSION CONTEXT DATASET CREATION
    print(f"4. Creation of FUSION context dataset with points {context_points}...\n")
    
    if context_points == "train":
        print(f"CXR CONTEXT DATASET ARGS:\n{cxr_context_hypers}")
        print()
        print(f"EHR CONTEXT DATASET ARGS:\n{ehr_context_hypers}")
        print()

        # ------------------------------------------------
        # EHR CONTEXT DATASET

        discretizer = get_discretizer(mimic_fusion_args)
        normalizer = get_normalizer(discretizer, mimic_fusion_args)
        context_dataset_ehr, _, _ = ehr_dataset.get_datasets(discretizer, normalizer, mimic_fusion_args, transform=True, transform_hypers=ehr_context_hypers)
        # ------------------------------------------------
        
        # ------------------------------------------------
        # CXR CONTEXT DATASET 
        DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
        DATA_STD = np.array([0.247, 0.243, 0.261])

        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img  
        
        gaussian_blur_kernel = (cxr_context_hypers["gaussianblur_kernel"], cxr_context_hypers["gaussianblur_kernel"])
        
        context_transform_list = [
            transforms.RandomCrop(size=cxr_context_hypers["randcrop_size"], padding=4),
            transforms.RandomHorizontalFlip(cxr_context_hypers["randhorizontalflip_prob"]), 
            transforms.RandomVerticalFlip(cxr_context_hypers["randverticalflip_prob"]),
            transforms.GaussianBlur(kernel_size=gaussian_blur_kernel),
            transforms.RandomSolarize(threshold=cxr_context_hypers["randsolarize_threshold"], p=cxr_context_hypers["randsolarize_prob"]),
            transforms.RandomInvert(cxr_context_hypers["randinvert_prob"]),
            transforms.ColorJitter(brightness=cxr_context_hypers["colorjitter_brightness"], contrast=cxr_context_hypers["colorjitter_contrast"]),
            transforms.Resize(224), # We need 224x224 for MIMIC-CXR
            ]
        
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
        context_dataset_cxr = cxr_dataset.MIMICCXR(cxr_paths, mimic_fusion_args, split="train", transform=context_transform)

        context_dataset, _, _ = fusion_dataset.load_cxr_ehr(mimic_fusion_args, context_dataset_ehr, ehr_val_ds, context_dataset_cxr, cxr_val_ds, ehr_test_ds, cxr_test_ds)
        context_dataset_size = len(context_dataset)
        # ------------------------------------------------
   
    elif context_points == "misclassified":
        if context_points_data_file == "":
            raise ValueError("You selected misclassified points for your context set but did not provide a data points file.")
        
        context_fusion_args = {
            "task": mimic_task, 
            "points": context_points,
            "data_points_file": context_points_data_file,
            "corrupted": corrupted,
            "merged": False,
            "ehr_corrupt_hypers": ehr_context_hypers,
            "cxr_corrupt_hypers": cxr_context_hypers
        }

        context_dataset = context_dataset_class.load_context_dataset(context_fusion_args)
        context_dataset_size = len(context_dataset)
    
    elif context_points == "latent":
        if context_points_data_file == "":
            raise ValueError("You selected latent points for your context set but did not provide a data points file.")
        
        context_fusion_args = {
            "task": mimic_task, 
            "points": context_points,
            "data_points_file": context_points_data_file,
            "corrupted": corrupted,
            "merged": False,
            "ehr_corrupt_hypers": ehr_context_hypers,
            "cxr_corrupt_hypers": cxr_context_hypers
        }

        context_dataset = context_dataset_class.load_context_dataset(context_fusion_args)
        context_dataset_size = len(context_dataset)
    
    elif context_points == "merged":
        print(f" Context set 1: Creation of corrupted train points...")
        print()
        # ------------------------------------------------
        # EHR CONTEXT DATASET

        discretizer = get_discretizer(mimic_fusion_args)
        normalizer = get_normalizer(discretizer, mimic_fusion_args)
        context_dataset_ehr, _, _ = ehr_dataset.get_datasets(discretizer, normalizer, mimic_fusion_args, transform=True, transform_hypers=ehr_context_hypers)
        # ------------------------------------------------
        
        # ------------------------------------------------
        # CXR CONTEXT DATASET 
        DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
        DATA_STD = np.array([0.247, 0.243, 0.261])

        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img  
        
        gaussian_blur_kernel = (cxr_context_hypers["gaussianblur_kernel"], cxr_context_hypers["gaussianblur_kernel"])
        
        context_transform_list = [
            transforms.RandomCrop(size=cxr_context_hypers["randcrop_size"], padding=4),
            transforms.RandomHorizontalFlip(cxr_context_hypers["randhorizontalflip_prob"]), 
            transforms.RandomVerticalFlip(cxr_context_hypers["randverticalflip_prob"]),
            transforms.GaussianBlur(kernel_size=gaussian_blur_kernel),
            transforms.RandomSolarize(threshold=cxr_context_hypers["randsolarize_threshold"], p=cxr_context_hypers["randsolarize_prob"]),
            transforms.RandomInvert(cxr_context_hypers["randinvert_prob"]),
            transforms.ColorJitter(brightness=cxr_context_hypers["colorjitter_brightness"], contrast=cxr_context_hypers["colorjitter_contrast"]),
            transforms.Resize(224), # We need 224x224 for MIMIC-CXR
            ]
        
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
        context_dataset_cxr = cxr_dataset.MIMICCXR(cxr_paths, mimic_fusion_args, split="train", transform=context_transform)

        context_dataset_1, _, _ = fusion_dataset.load_cxr_ehr(mimic_fusion_args, context_dataset_ehr, ehr_val_ds, context_dataset_cxr, cxr_val_ds, ehr_test_ds, cxr_test_ds)

        print(f"\n Context set 2: Loading context set file...")
        if context_points_data_file == "":
            raise ValueError("You selected merged points for your context set but did not provide a data points file to merge with corrupted train points.")
        
        context_fusion_args = {
            "task": mimic_task, 
            "points": context_points,
            "data_points_file": context_points_data_file,
            "corrupted": corrupted,
            "merged": True,
            "ehr_corrupt_hypers": ehr_context_hypers,
            "cxr_corrupt_hypers": cxr_context_hypers
        }

        context_dataset_2 = context_dataset_class.load_context_dataset(context_fusion_args)
        
        context_dataset = data.ConcatDataset([context_dataset_1, context_dataset_2])
        
        context_dataset_size = len(context_dataset)

        print(f" Merged context set length: {context_dataset_size}")
    else:
        ValueError("Unknown context dataset type.")

    num_context_batches = context_dataset_size/context_batch_size
    num_training_batches = training_dataset_size/batch_size

    if num_context_batches < num_training_batches:
        print(" \n Extending context set size to match number of training batches")
        context_dataset = CustomDataset(context_dataset, training_dataset_size*int(context_batch_size/batch_size))
        context_dataset_size = len(context_dataset)
    
    print(f"Dataset length\n Context: {context_dataset_size}")
    print("FUSION CONTEXT dataset done.")
    print()
    # ------------------------------------------------------------------------

    print(f"Fusion datasets sizes\n Training: {training_dataset_size} | Validation: {val_dataset_size} | Testing: {test_dataset_size} | Context: {context_dataset_size}")
    print(" Finished creating FUSION-CXR-EHR datasets")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print()

else:
    raise ValueError("Dataset selected not found.")
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# DATALOADERS
# We use a different collate function for the dataloaders on the MIMIC-EHR and MIMIC-FUSION datasets.
    
if dataset == "mimic-fusion" or ((dataset == "mimic-cxr" or dataset == "mimic-ehr") and (mimic_task == "in-hospital-mortality" or mimic_task == "phenotyping")):
    # Set up num_works depending on your available hardware
    num_workers = 16

    if context_points in ["train", "merged"]:
        context_collate_fn = fusion_dataset.my_collate
    elif context_points in ["misclassified", "latent"]:
        context_collate_fn = context_dataset_class.my_collate
    else:
        # Any other scenario we may not consider
        context_collate_fn = fusion_dataset.my_collate

    train_loader = data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=fusion_dataset.my_collate,
                                    num_workers=num_workers,
                                    persistent_workers=True)
    
    val_loader = data.DataLoader(validation_dataset, 
                                    batch_size=batch_size,
                                    shuffle=False,
                                    collate_fn=fusion_dataset.my_collate,
                                    num_workers=num_workers,
                                    persistent_workers=True)
    
    test_loader  = data.DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    collate_fn=fusion_dataset.my_collate,
                                    num_workers=num_workers,
                                    persistent_workers=True)
    
    context_loader  = data.DataLoader(context_dataset,
                                    batch_size=context_batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    collate_fn=context_collate_fn,
                                    num_workers=num_workers,
                                    persistent_workers=True)

else:
    raise ValueError("No dataset correctly selected. No dataloaders generated.")
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# JAX TRAINER
class TrainState(train_state.TrainState):
    batch_stats: Any
    # params_logvar: Any

class TrainerModule:
    def __init__(self,
                 model_name : str,
                 model_class : nn.Module,
                 optimizer_name : str,
                 model_hparams : dict,
                 optimizer_hparams : dict,
                 objective_hparams : dict,
                 ssm_hparams: dict,
                 other_hparams: dict,
                 exmp_inputs : Any,
                 ):
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.optimizer_name = optimizer_name

        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.ssm_hparams = ssm_hparams

        # Objective hyperparams
        self.prior_mean = objective_hparams["prior_mean"]
        self.prior_var = objective_hparams["prior_var"]
        self.objective_hparams = objective_hparams
        self.mc_samples_llk = objective_hparams["mc_samples_llk"]
        self.mc_samples_reg = objective_hparams["mc_samples_reg"]
        self.training_dataset_size = objective_hparams["training_dataset_size"]
        self.batch_size = objective_hparams["batch_size"]
        self.prior_likelihood_scale = objective_hparams["prior_likelihood_scale"]
        self.prior_likelihood_f_scale = objective_hparams["prior_likelihood_f_scale"]
        self.prior_likelihood_cov_scale = objective_hparams["prior_likelihood_cov_scale"]
        self.prior_likelihood_cov_diag = objective_hparams["prior_likelihood_cov_diag"]
        self.prior_likelihood_mean = objective_hparams["prior_likelihood_mean"]
        self.prior_likelihood_normalize_feature = objective_hparams["prior_likelihood_normalize_feature"]
        self.likelihood_scale = objective_hparams["likelihood_scale"]
        self.reg_scale = objective_hparams["reg_scale"]
        self.reg_scale_gap = objective_hparams["reg_scale_gap"]
        self.gap_regularizer = objective_hparams["gap_regularizer"]
        self.rho_sam = objective_hparams["rho_sam"]
        self.reg_type = self.objective_hparams["reg_type"]
        self.init_logvar = objective_hparams["init_logvar"]
        self.init_final_layer_weights_logvar = objective_hparams["init_final_layer_weights_logvar"]
        self.init_final_layer_bias_logvar = objective_hparams["init_final_layer_bias_logvar"]
        self.prior_feature_logvar = objective_hparams["prior_feature_logvar"]
        self.pretrained_prior = objective_hparams["pretrained_prior"]
        self.det_model_fine_tuning = objective_hparams["det_model_fine_tuning"]
        
        # Other hyperparams
        self.other_hparams = other_hparams
        self.stochastic = other_hparams["stochastic"]
        self.set_zero_gradients = other_hparams["set_zero_gradients"]
        self.jobid = other_hparams["jobid"]
        self.seed = other_hparams["seed"]
        self.num_epochs = other_hparams["num_epochs"]
        self.evaluate = other_hparams["evaluate"]
        self.build_context_set = other_hparams["build_context_set"]
        self.linearize = other_hparams['linearize']
        self.restore_checkpoint = other_hparams["restore_checkpoint"]
        self.batch_stats_init_epochs = other_hparams["batch_stats_init_epochs"]
        self.mc_samples_eval = other_hparams["mc_samples_eval"]
        self.dataset = other_hparams["dataset"]
        self.output_var = other_hparams["output_var"]
        self.prediction_type = other_hparams["prediction_type"]
        self.mimic_task = other_hparams["mimic_task"]
        self.debug_print = other_hparams["debug_print"]
        self.debug_print_updated = other_hparams["debug_print"]
        self.log_frequency = other_hparams["log_frequency"]
        self.final_eval = other_hparams["final_eval"]
        self.eval_patient_groups = other_hparams["eval_patient_groups"]
        self.save_to_wandb = other_hparams["save_to_wandb"]
        self.save_model_on = other_hparams["save_model_on"]
        self.wandb_project = other_hparams["wandb_project"]
        self.wandb_account = other_hparams["wandb_account"]
        self.confidence_intervals = other_hparams["confidence_intervals"]
        self.label_file_splits = other_hparams["label_file_splits"]
        self.full_training = other_hparams["full_training"]
        self.model_for_final_eval = other_hparams["model_for_final_eval"]
        self.pretrained_prior_path = other_hparams["pretrained_prior_path"]
        self.save_plot_data = other_hparams["save_plot_data"]

        self.n_batches_train = self.training_dataset_size / self.batch_size
        self.num_classes = self.model_hparams["num_classes"]
        self.inference_data_model_name = f"{inference_data_model_name}_{self.mimic_task}_seed_{self.seed}"

        self.n_batches_eval = 100
        self.n_batches_eval_context = 10
        self.n_batches_eval_final = 100

        self.params_prior_mean = None
        self.params_prior_logvar = None
        self.batch_stats_prior = None
        self.pred_fn = None

        # --------------------------------------------------------------------
        # MODEL CONDITIONS
        if self.model_name == "ResNet18-Pretrained":
            self.model = ResNet18(output="logits", pretrained="imagenet", num_classes=self.num_classes, dtype="float32")
        elif self.model_name == "ResNet34": # We use this in uni-modal MIMIC CXR
            print("Model for UNI-CXR")
            self.model = ResNet34(output="logits", pretrained="imagenet", num_classes=self.num_classes, dtype="float32")
        elif self.model_name == "ResNet50-Pretrained":
            self.model = ResNet50(output="logits", pretrained="imagenet", num_classes=self.num_classes, dtype="float32")
        elif self.model_name == "LSTM": # We use this in uni-modal MIMIC EHR
            print("Model for UNI-EHR")
            self.model = lstm_jax.LSTM(fusion=False, input_dim=76, num_classes=self.num_classes, feats_dim=128, batch_first=True, dropout=0.3, layers=2)
        elif self.model_name == "Fusion": # We use this in multi-modal MIMIC CXR-EHR
            if self.mimic_task == "phenotyping":
                self.model = fusion_jax.Fusion(num_classes=25, vision_num_classes=25, labels_set="phenotyping")
            elif self.mimic_task == "in-hospital-mortality":
                self.model = fusion_jax.Fusion(num_classes=1, vision_num_classes=1, labels_set="mortality")
        else:
            self.model = self.model_class(**self.model_hparams)
        # --------------------------------------------------------------------


        # --------------------------------------------------------------------
        # TRAINING METHOD BAYESIAN OR DETERMINISTIC
        # We add this to be used as a unique identifier for each experiment
        # based on the time at which training was performed.
        # This is a second safety measure to ensure that we load the correct 
        # model weights for test evaluation.
        exprt_time = str(time.time()).split(".")[-1] 

        if method == "psvi":
            self.run_name = f"{exprt_time}_{model_name}_BNN_{num_epochs}_{learning_rate}_{reg_type}_ctxtbsize:{context_batch_size}_prlkcovs_{prior_likelihood_cov_scale}_exprt_{jobid}_seed_{seed}_{self.mimic_task}_{self.label_file_splits}"
        if method == "psmap":
            self.run_name = f"{exprt_time}_{model_name}_DET_{num_epochs}_{learning_rate}_{batch_size}_{alpha}_{exponent}_exprt_{jobid}_seed_{seed}_{self.mimic_task}_{self.label_file_splits}"
        if self.full_training:
            self.run_name += "_full_training"
            self.save_model_on = "None"
        if self.gap_regularizer:
            self.run_name += f"_gap_{self.reg_scale_gap}_rhos_{self.rho_sam}"
        # --------------------------------------------------------------------


        # --------------------------------------------------------------------
        # TRAINING LOGGER
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.run_name)
        if self.prediction_type == "mimic":
            self.logger = {
            "epoch": [],
            "loss_train": [],
            "loss_val": [],
            "loss_test": [0],
            "acc_train": [],

            # AUROC
            "auroc_test": [0],
            "auroc_ci_test": [],
            "auroc_val": [],
            "auroc_ci_val": [],
            "auroc_val_best": [],
            "auroc_ci_val_best": [],
            "auroc_selective_test":[0],
            "auroc_selective_val":[],
            "auroc_selective_val_best":[],

            # AUPRC
            "auprc_test": [0],
            "auprc_ci_test": [],
            "auprc_val": [],
            "auprc_ci_val": [],
            "auprc_val_best": [],
            "auprc_ci_val_best": [],
            "auprc_selective_test":[0],
            "auprc_selective_val":[],

            "auroc_auprc_val": [],
            "auroc_auprc_val_best": [],
            "auroc_auprc_test": [0],

            # NLL
            "nll_test": [0],
            "nll_val": [],
            "nll_median_test": [0],
            "nll_median_val": [],
            "nll_sum_test": [0],
            "nll_sum_val": [],
            
            # ECE
            "ece_test": [0],
            "ece_val": []
            }
        else:
            self.logger = {
                "epoch": [],
                "loss_train": [],
                "acc_train": [],
                "nll_test": [],
                "ece_test": [],
                "acc_test": [],
                "acc_test_best": [],
                "acc_sel_test": [],
                "acc_sel_test_ood": [],
                "ood_auroc_entropy": [],
                "ood_auroc_aleatoric": [],
                "ood_auroc_epistemic": [],
                "predictive_entropy_test": [],
                "aleatoric_uncertainty_test": [],
                "epistemic_uncertainty_test": [],
                "predictive_entropy_context": [],
                "aleatoric_uncertainty_context": [],
                "epistemic_uncertainty_context": [],
                "predictive_entropy_ood": [],
                "aleatoric_uncertainty_ood": [],
                "epistemic_uncertainty_ood": [],
            }

        if "cifar10" in self.dataset and "cifar100" not in self.dataset:
            self.logger["acc_test_cifar101"] = []
            self.logger["acc_sel_test_cifar101"] = []
            self.logger["nll_test_cifar101"] = []
            self.logger["ece_test_cifar101"] = []
        if self.final_eval:
            if "cifar10" in self.dataset and "cifar100" not in self.dataset:
                for corr_config in corr_config_list:
                    self.logger[f"acc_test_ccifar10_{corr_config}"] = []
                    self.logger[f"acc_test_ccifar10_{corr_config}"] = []
                    self.logger[f"acc_sel_test_ccifar10_{corr_config}"] = []
        self.wandb_logger = []
        # --------------------------------------------------------------------

        self.create_functions()
        if (self.dataset == "mimic-cxr" or self.dataset == "mimic-ehr") and (self.mimic_task == "in-hospital-mortality" or self.mimic_task == "phenotyping"):
            if self.dataset == "mimic-cxr":
                self.init_model(exmp_inputs[2])
            elif self.dataset == "mimic-ehr":
                self.init_model(exmp_inputs[0])
        else:
            self.init_model(exmp_inputs)
        # print(self.model.tabulate(random.PRNGKey(0), x=exmp_inputs[0]))

        assert self.mc_samples_llk == 1 if not self.stochastic else True
        assert self.mc_samples_eval == 1 if not self.stochastic else True
        assert self.mc_samples_reg == 1 # if not ("fsmap" in method or "fsvi" in method) else True  # currently not implemented
        # assert self.objective_hparams["reg_points"] == "train" if self.objective_hparams["method"] == "psmap" else True
    
    def create_functions(self):
        def calculate_cov(jac, logvar):
            var = jnp.exp(logvar)
            # jac has shape (batch_dim, output_dim, params_dims...)
            # jac_2D has shape (batch_dim * output_dim, nb_params)
            batch_dim, output_dim = jac.shape[:2]
            jac_2D = jnp.reshape(jac, (batch_dim * output_dim, -1))
            # sigma_flatten has shape (nb_params,) and will be broadcasted to the same shape as jac_2D
            sigma_flatten = jnp.reshape(var, (-1,))
            # jac_sigma_product has the same shape as jac_2D
            jac_sigma_product = jnp.multiply(jac_2D, sigma_flatten)
            cov = jnp.matmul(jac_sigma_product, jac_2D.T)
            cov = jnp.reshape(cov, (batch_dim, output_dim, batch_dim, output_dim))
            return cov

        def calculate_moments(params_mean, params_logvar, inputs, batch_stats, rng_key):
            ### Split both mean and logvar parameters
            params_feature_mean, params_final_layer_mean = split_params(params_mean, "dense")
            params_feature_logvar, params_final_layer_logvar = split_params(params_logvar, "dense")

            ### sample feature parameters and merge with final-layer mean parameters
            params_feature_sample = sample_parameters(params_feature_mean, params_feature_logvar, self.stochastic, rng_key)
            params_partial_sample = merge_params(params_feature_sample, params_final_layer_mean)

            ### feature covariance (mostly the same as Jacobian covariance (does not include bias term), up to numerical errors)
            _out = self.model.apply({'params': params_partial_sample, 'batch_stats': batch_stats},
                                    inputs,
                                    train=True,
                                    feature=True,
                                    mutable=['batch_stats'])
            out, _ = _out
            preds_f_sample, feature_sample = out[0], out[1]
            
            n_samples = preds_f_sample.shape[0]
            feature_dim = params_final_layer_mean[self.final_layer_key]["kernel"].shape[0]
            final_layer_var_weights = jnp.exp(params_final_layer_logvar[self.final_layer_key]["kernel"])
            final_layer_var_bias = jnp.exp(params_final_layer_logvar[self.final_layer_key]["bias"])

            feature_times_var = (jnp.repeat(final_layer_var_weights, n_samples).reshape(n_samples, feature_dim, self.num_classes) * feature_sample[:, :, None]).transpose(2, 0, 1)
            preds_f_cov = jnp.matmul(feature_times_var, feature_sample.T).transpose(1, 2, 0)
            preds_f_cov += preds_f_cov + final_layer_var_bias[None, None, :]

            return preds_f_sample, preds_f_cov

        def calculate_function_kl(params_variational_mean, params_variational_logvar, inputs, batch_stats, rng_key):
            ### set prior batch stats
            if self.batch_stats_init_epochs == 0:
                batch_stats_prior = jax.lax.stop_gradient(batch_stats)
            else:
                batch_stats_prior = self.batch_stats_prior
                
            ### set prior mean parameters
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                # params_prior_mean = jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"])
                params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
                # params_prior_mean = jax.tree_map(lambda x, y: x + y, jax.lax.stop_gradient(params), jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"]))

            ### set parameter prior variance
            feature_prior_logvar = self.prior_feature_logvar
            final_layer_prior_logvar = jnp.log(self.prior_var)

            ### initialize and split prior logvar parameters into feature and final-layer parameters
            params_prior_logvar_init = jax.tree_map(lambda x: x * 0, params_prior_mean)  # initialize logvar parameters with zeros
            params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = split_params(params_prior_logvar_init, "dense")

            ### set feature and final-layer logvar parameters separately
            params_feature_prior_logvar = jax.tree_map(lambda x: x * 0 + feature_prior_logvar, params_feature_prior_logvar_init)
            params_final_layer_prior_logvar = jax.tree_map(lambda x: x * 0 + final_layer_prior_logvar, params_final_layer_prior_logvar_init)

            ### merge logvar parameters
            params_prior_logvar = merge_params(params_feature_prior_logvar, params_final_layer_prior_logvar)

            preds_f_prior_mean, preds_f_prior_cov = calculate_moments(params_prior_mean, params_prior_logvar, inputs, batch_stats_prior, rng_key)
            preds_f_variational_mean, preds_f_variational_cov = calculate_moments(params_variational_mean, params_variational_logvar, inputs, batch_stats, rng_key)

            if self.debug_print_updated:
                jax.debug.print("\ncov prior:\n{}", preds_f_prior_cov[0:2, 0, 0:2, 0])
                jax.debug.print("cov variational:\n{}\n", preds_f_variational_cov[0:2, 0, 0:2, 0])
                jax.debug.print("cov prior inv:\n{}", jnp.linalg.inv(preds_f_prior_cov)[0:2, 0, 0:2, 0])
                jax.debug.print("cov variational inv:\n{}\n", jnp.linalg.inv(preds_f_variational_cov)[0:2, 0, 0:2, 0])

            kl = 0
            n_samples = preds_f_variational_mean.shape[0]
            cov_jitter = 1e-6
            for j in range(self.num_classes):
                _preds_f_prior_mean = preds_f_prior_mean[:, j].transpose()
                _preds_f_prior_cov = preds_f_prior_cov[:, :, j]  # + jnp.eye(n_samples) * cov_jitter

                _preds_f_variational_mean = preds_f_variational_mean[:, j].transpose()
                _preds_f_variational_cov = preds_f_variational_cov[:, :, j] + jnp.eye(n_samples) * cov_jitter

                q = tfd.MultivariateNormalFullCovariance(
                    loc=_preds_f_variational_mean,
                    covariance_matrix=_preds_f_variational_cov,
                    validate_args=False,
                    allow_nan_stats=True,
                )
                p = tfd.MultivariateNormalFullCovariance(
                    loc=_preds_f_prior_mean,
                    covariance_matrix=_preds_f_prior_cov,
                    validate_args=False,
                    allow_nan_stats=True,
                )
                kl += tfd.kl_divergence(q, p, allow_nan_stats=False)

            return kl

        def calculate_function_prior_density(preds_f, params, inputs, batch_stats, rng_key, prior_var):
            ### set prior batch stats
            if self.batch_stats_init_epochs == 0:
                batch_stats_prior = jax.lax.stop_gradient(batch_stats)
            else:
                batch_stats_prior = self.batch_stats_prior

            ### set parameter prior mean
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                # params_prior_mean = jax.lax.stop_gradient(self.model.init(rng_key, inputs[0:1], train=True)["params"])
                params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
                # params_prior_mean = jax.lax.stop_gradient(params)
            params_feature_prior_mean, params_final_layer_prior_mean = split_params(params_prior_mean, "dense")

            ### initialize and split prior logvar parameters into feature and final-layer parameters
            params_prior_logvar_init = jax.tree_map(lambda x: x * 0, params_prior_mean)  # initialize logvar parameters with zeros
            params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = split_params(params_prior_logvar_init, "dense")

            ### set feature parameter logvar and final-layer parameter variance
            feature_prior_logvar = self.prior_feature_logvar
            final_layer_prior_logvar = jnp.log(prior_var)
            params_feature_prior_logvar = jax.tree_map(lambda x: x * 0 + feature_prior_logvar, params_feature_prior_logvar_init)
            params_final_layer_prior_logvar = jax.tree_map(lambda x: x * 0 + final_layer_prior_logvar, params_final_layer_prior_logvar_init)

            params_feature_prior_sample = sample_parameters(params_feature_prior_mean, params_feature_prior_logvar, self.stochastic, rng_key)
            
            params_prior_sample = merge_params(params_feature_prior_sample, params_final_layer_prior_mean)

            ### feature covariance (mostly the same as Jacobian covariance, up to numerical errors)
            _out = self.model.apply({'params': params_prior_sample, 'batch_stats': batch_stats_prior},
                                    inputs,
                                    train=True,
                                    feature=True,
                                    mutable=['batch_stats'])
            out, _ = _out

            preds_f_prior_mean, feature_prior = jax.lax.stop_gradient(out[0]), jax.lax.stop_gradient(out[1])
            
            preds_f_prior_cov = prior_var * jnp.matmul(feature_prior, feature_prior.T)  # assumes the prior is identical across output dimensions
            preds_f_prior_cov += jnp.ones_like(preds_f_prior_cov) * prior_var  # add bias variance
            preds_f_prior_cov += jnp.eye(preds_f_prior_cov.shape[0]) * prior_var  # jnp.max(jnp.array([eps_cov * prior_var, eps_cov]))  # add small constant to the diagonal to ensure positive definiteness

            p = tfd.MultivariateNormalFullCovariance(
                loc=preds_f_prior_mean[:, 0],  # assumes the prior is identical across output dimensions
                covariance_matrix=preds_f_prior_cov,
                validate_args=False,
                allow_nan_stats=True,
            )
            log_density = jnp.sum(p.log_prob(preds_f[0].T))

            reg = -log_density

            if self.debug_print_updated:
                jax.debug.print("\nf - mean: {}", jnp.mean(preds_f[:, 0] - preds_f_prior_mean[:, 0]))
                jax.debug.print("log_density: {}\n", p.log_prob(preds_f.T))
                jax.debug.print("cholesky: {}\n", jnp.linalg.cholesky(preds_f_prior_cov[:, 0, :, 0]))

            return reg

        # USED IN MEDFUSE
        def calculate_empirical_gaussian_prior_kl(params_samples, params_variational_mean, params_variational_logvar, preds_f, inputs, batch_stats, prior_likelihood_scale, prior_likelihood_f_scale, prior_likelihood_cov_scale, prior_likelihood_cov_diag, rng_key):
            # Fusion model requires specific rng_key with dropout key
            if self.prediction_type == "mimic":
                root_key = jax.random.PRNGKey(self.seed)
                params_rng, dropout_key  = random.split(root_key, num=2)
                rng_key = {'params': params_rng, 'dropout': dropout_key} # Required for the dropout layer in LSTM of Fusion model
            else:
                inputs = inputs[0:1] # Specific inputs for all the other cases. In Fusion we send the whole tuple of inputs/labels

            ### set prior batch stats
            if self.batch_stats_init_epochs == 0:
                batch_stats_prior = jax.lax.stop_gradient(batch_stats)
            else:
                batch_stats_prior = self.batch_stats_prior

            ### set parameter prior mean
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                params_prior_mean = jax.tree_map(lambda x: x * prior_likelihood_scale, jax.lax.stop_gradient(self.model.init(rng_key, inputs, train=True)["params"]))
                # params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
                # params_prior_mean = jax.lax.stop_gradient(params)

            _out = self.model.apply({'params': params_prior_mean, 'batch_stats': batch_stats_prior},
                                    inputs,
                                    train=True,
                                    feature=True,
                                    mutable=['batch_stats'],
                                    rngs=rng_key
                                    )
            out, _ = _out
            preds_f_prior_mean, feature_prior = jax.lax.stop_gradient(out[0]), jax.lax.stop_gradient(out[1])
            
            cross_entropy = 0
            for i, params_sample in enumerate(params_samples):
                _out = self.model.apply({'params': params_sample, 'batch_stats': batch_stats},
                                        inputs,
                                        train=True,
                                        feature=True,
                                        mutable=['batch_stats'],
                                        rngs=rng_key
                                        )
                out, _ = _out
                feature_variational_sample = out[1]

                if self.prior_likelihood_normalize_feature:
                    feature_cov = (feature_variational_sample - feature_variational_sample.mean(0)) / feature_variational_sample.std(0)
                else:
                    feature_cov = feature_variational_sample
                    # feature_cov = feature_prior

                n_inputs = feature_cov.shape[0]

                if self.prediction_type == "mimic":
                    # TODO Do we actually need to fix it?? in a cleaner, more elegant way?? this comes from init_model
                    # The problem here is the repeat function is not in line the the dimensions of the reshape
                    # self.final_layer_key = ["fused_cls"]["layers_0"]
                    feature_dim = params_variational_logvar["fused_cls"]["layers_0"]["kernel"].shape[0]
                    final_layer_var_weights = jnp.exp(params_variational_logvar["fused_cls"]["layers_0"]["kernel"])
                    final_layer_var_bias = jnp.exp(params_variational_logvar["fused_cls"]["layers_0"]["bias"])
                else:
                    feature_dim = params_variational_logvar[self.final_layer_key]["kernel"].shape[0]
                    final_layer_var_weights = jnp.exp(params_variational_logvar[self.final_layer_key]["kernel"])
                    final_layer_var_bias = jnp.exp(params_variational_logvar[self.final_layer_key]["bias"])

                
                # feature_times_var = (jnp.repeat(final_layer_var_weights, n_inputs).reshape(n_inputs, feature_dim, self.num_classes) * feature_variational_sample[:, :, None]).transpose(2, 0, 1)
                # preds_f_variational_cov = jnp.matmul(feature_times_var, feature_variational_sample.T).transpose(1, 2, 0)
                # preds_f_variational_cov += preds_f_variational_cov + final_layer_var_bias[None, None, :]
                # preds_f_variational_cov += jnp.eye(preds_f_variational_cov.shape[0])[:, :, None] * (1e-1 + prior_likelihood_cov_scale)  # add jitter for numerical stability

                feature_times_var = (jnp.repeat(final_layer_var_weights, n_inputs).reshape(n_inputs, feature_dim, self.num_classes) * feature_cov[:, :, None]).transpose(2, 0, 1)
                preds_f_variational_cov = jnp.matmul(feature_times_var, feature_cov.T).transpose(1, 2, 0)
                preds_f_variational_cov += preds_f_variational_cov + final_layer_var_bias[None, None, :]
                preds_f_variational_cov += prior_likelihood_cov_diag * jnp.eye(preds_f_variational_cov.shape[0])[:, :, None]  # add jitter for numerical stability
                preds_f_variational_cov *= prior_likelihood_cov_scale * preds_f_variational_cov

                _cross_entropy = 0
                for j in range(self.num_classes):

                    if self.mimic_task == "phenotyping" and self.set_zero_gradients:
                        # Selected list of labels based on analysis of per-label results
                        if j in [0, 1, 3, 4, 7, 8, 9, 10, 13, 14, 21, 23, 24, 22]:
                            preds_f[i, :, j] = 0
                    
                    q = tfd.MultivariateNormalFullCovariance(
                        loc=preds_f[i, :, j],
                        covariance_matrix=preds_f_variational_cov[:, :, j],
                        validate_args=False,
                        allow_nan_stats=True,
                    )

                    _cross_entropy += -(1 / self.num_classes) * jnp.sum(q.log_prob(prior_likelihood_f_scale * preds_f_prior_mean[:, j].T))

                cross_entropy += (1 / len(params_samples)) * _cross_entropy
                # cross_entropy += 1000 * jnp.sum(jnp.square(final_layer_var_weights.flatten() - prior_likelihood_cov_scale)) + jnp.sum(jnp.square(final_layer_var_bias.flatten() - prior_likelihood_cov_scale))
                
            kl_params = calculate_parameter_kl(params_variational_mean, params_variational_logvar)

            kl = kl_params + cross_entropy

            if self.debug_print_updated:
                jax.debug.print("kl_params: {}", kl_params)
                jax.debug.print("kl_prior: {}", cross_entropy)

            return kl

        def calculate_empirical_categorical_prior(preds_f, prediction_type):
            if prediction_type == "classification":
                log_preds = jax.nn.log_softmax((1 / self.prior_likelihood_scale) * preds_f, -1)
                cross_entropy = jnp.sum(-(1 / self.num_classes) * log_preds, axis=-1)  # sum over output dimensions
            elif prediction_type == "regression":
                preds_f_std = jnp.std(preds_f, axis=0)
                cross_entropy = jnp.sum(jnp.log(preds_f_std * jnp.sqrt(2 * jnp.pi * jnp.e)), axis=-1)  # sum over output dimensions

            reg = jnp.mean(jnp.sum(cross_entropy, axis=-1), axis=0)  # sum over input points and contract samples dimension

            return reg
        
        def calculate_empirical_categorical_prior_kl(params_samples, params_variational_logvar, preds_f, prior_var, prediction_type):
            if prediction_type == "classification":
                log_preds = jax.nn.log_softmax((1 / self.prior_likelihood_scale) * preds_f, -1)
                cross_entropy = jnp.sum(-(1 / self.num_classes) * log_preds, axis=-1)  # sum over output dimensions
                cross_entropy = jnp.mean(jnp.sum(cross_entropy, axis=-1), axis=0)  # sum over input points and take mean over MC samples
            elif prediction_type == "regression":
                # NotImplementedError
                preds_f_std = jnp.std(preds_f, axis=0)  # computed over MC samples
                cross_entropy = jnp.sum(jnp.log((preds_f_std + self.prior_likelihood_scale) * jnp.sqrt(2 * jnp.pi * jnp.e)), axis=-1)  # sum over output dimensions
                cross_entropy = jnp.sum(cross_entropy, axis=-1)  # sum over input points

            for params_sample in params_samples:
                cross_entropy += (1 / self.mc_samples_llk) * calculate_parameter_norm(params_sample, prior_var)
            
            ### remove batchnorm parameters from the KL calculation
            # params_variational_logvar, _ = split_params(params_variational_logvar, "batch_norm")

            params_variational_var = jax.tree_map(lambda x: jnp.exp(x), params_variational_logvar)

            neg_entropy = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(
                lambda x: -jnp.log((x ** 0.5) * jnp.sqrt(2 * jnp.pi * jnp.e)), params_variational_var))[0]
                )

            kl = neg_entropy + cross_entropy

            return kl

        def calculate_entropic_prior(preds_f):
            preds = jax.nn.softmax(preds_f, -1)
            entropy = jnp.sum(preds * jnp.log(preds + 1e-10), axis=-1)
            reg = jnp.mean(self.prior_likelihood_scale * entropy)

            return reg

        def calculate_function_norm(preds_f_reg, inputs, batch_stats, prior_var):
            if self.params_prior_mean is None:
                preds_f_prior_mean = jnp.zeros_like(preds_f_reg)
            else:
                _out = self.model.apply({'params': self.params_prior_mean, 'batch_stats': batch_stats},
                                        inputs,
                                        train=True,
                                        mutable=['batch_stats'])
                out, _ = _out
                preds_f_prior_mean = jax.lax.stop_gradient(out)
            reg = 1 / (2 * prior_var) * jnp.sum(jnp.square(preds_f_reg - preds_f_prior_mean))  # 1/(2 * function_var) * ||f(inputs, params) - f(inputs, params_prior_mean)||^2

            return reg

        def calculate_parameter_norm(params, prior_var):
            params_model = params
            # params_model, params_batchnorm = split_params(params, "batch_norm")

            if self.objective_hparams["reg_type"] != "feature_parameter_norm":
                params_reg = params_model
                if self.params_prior_mean is None:
                    params_reg_prior_mean = jax.tree_map(lambda x: x * 0, jax.lax.stop_gradient(params_model))
                else:
                    params_reg_prior_mean = self.params_prior_mean
                reg = 1 / (2 * prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.square(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params - params_prior_mean||^2
            else:
                params_reg, _ = split_params(params_model, "dense")
                params_reg_prior_mean, _ = split_params(self.params_prior_mean, "dense")
                reg = 1 / (2 * prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.square(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params_feature - params_feature_prior_mean||^2

            return reg  # this scaling makes prior_precision consistent with the MAP objective scaling but inconsistent with the weight decay coefficient

        def calculate_sharpness_prior_gap(params, batch_stats, inputs, targets, rng_key, train=True):        
            def loss_fn(_params):
                out = self.pred_fn(
                    {'params': _params, 'batch_stats': batch_stats},
                    inputs, # all input points
                    train=train,
                    mutable=['batch_stats'] if train else False,
                    rngs={'dropout': rng_key}
                    )
                logits_f, new_model_state = out if train else (out, None)

                preds_f = jax.nn.sigmoid(logits_f)

                # Cross-entropy loss without sum across patients
                loss_matrix = -(targets*jnp.log(preds_f + 1e-10) + (1 - targets)*jnp.log(1 - preds_f + 1e-10))
                
                # Mask out any patient-label pair that is not positive
                loss_matrix_masked =  loss_matrix*targets

                # Sum across patients and across labels
                loss = jnp.sum(loss_matrix_masked) / targets.sum()

                return loss, (preds_f, new_model_state)

            _, grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params)

            _epsilon = grads

            _epsilon = jax.lax.stop_gradient(_epsilon)

            epsilon = dual_vector(_epsilon)

            params_perturbed = jax.tree_map(lambda a, b: a + b * self.rho_sam, params, epsilon)

            out = self.pred_fn(
                {'params': params_perturbed, 'batch_stats': batch_stats},
                inputs, # Forward pass on all training points form the batch
                train=train,
                mutable=['batch_stats'] if train else False,
                rngs={'dropout': rng_key}
                )
            
            logits_f_reg, new_model_state_perturbed = out if train else (out, None)
            
            preds_f_reg = jax.nn.sigmoid(logits_f_reg)
        
            # Cross-entropy loss without sum across patients
            nll_perturbed_matrix = -(targets*jnp.log(preds_f_reg + 1e-10) + (1 - targets)*jnp.log(1 - preds_f_reg + 1e-10))
            
            # Mask out any patient-label pair that is not positive
            nll_perturbed_matrix_masked =  nll_perturbed_matrix*targets

            # Sum across patients and across labels
            nll_perturbed = jnp.sum(nll_perturbed_matrix_masked) / targets.sum()

            # Keep it fixed
            prior_likelihood_variance = 1 / self.n_batches_train

            sharpness_penalty = jnp.sum(nll_perturbed / prior_likelihood_variance)

            parameter_norm = calculate_parameter_norm(params, self.prior_var)

            reg = parameter_norm + sharpness_penalty

            return reg, new_model_state_perturbed

        def kl_univariate_gaussians(mean_q, var_q, mean_p, var_p):
            logstd_jitter = 0
            kl_1 = jnp.log((var_p + logstd_jitter) ** 0.5) - jnp.log((var_q + logstd_jitter) ** 0.5)
            kl_2 = ((var_q + logstd_jitter) + (mean_q - mean_p) ** 2) / (var_p + logstd_jitter)
            kl_3 = -1
            kl = 0.5 * (kl_1 + kl_2 + kl_3)

            return kl

        def calculate_parameter_kl(params_variational_mean, params_variational_logvar):
            if self.params_prior_mean is not None:
                params_prior_mean = self.params_prior_mean
                params_prior_logvar = self.params_prior_logvar
            else:
                params_prior_mean = jax.tree_map(lambda x: x * 0 + self.prior_mean, jax.lax.stop_gradient(params_variational_mean))
                params_prior_logvar = jax.tree_map(lambda x: x * 0 + jnp.log(self.prior_var), jax.lax.stop_gradient(params_variational_logvar))

            params_prior_var = jax.tree_map(lambda x: jnp.exp(x), params_prior_logvar)
            params_variational_var = jax.tree_map(lambda x: jnp.exp(x), params_variational_logvar)

            kl = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(
                lambda a, b, c, d: kl_univariate_gaussians(a, b, c, d),
                params_variational_mean, params_variational_var, params_prior_mean, params_prior_var
                ))[0])
            
            return kl

        # USED IN MEDFUSE
        def sample_parameters(params, params_logvar, stochastic, rng_key):
            if stochastic:
                eps = jax.tree_map(lambda x: random.normal(rng_key, x.shape), params_logvar)
                params_std_sample = jax.tree_map(lambda x, y: x * jnp.exp(y) ** 0.5, eps, params_logvar) # params_logvar comes as none
                params_sample = jax.tree_map(lambda x, y: x + y, params, params_std_sample) # Line giving us trouble because of params_logvar
            else:
                params_sample = params
            return params_sample
            
        def f_lin(params_dict, inputs, train, mutable):

            params = params_dict["params"]
            batch_stats = params_dict["batch_stats"]

            out = self.model.apply(
                {'params': jax.lax.stop_gradient(params), 'batch_stats': batch_stats},
                inputs,
                train=train,
                mutable=['batch_stats'] if train else False
                )
            _, new_model_state = out if train else (out, None)

            if train:
                _pred_fn = lambda params: self.model.apply(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=True,
                    mutable=['batch_stats']
                    )[0]
        
                eps = jax.tree_map(lambda x: random.normal(rng_key, x.shape), jax.lax.stop_gradient(params))
                eps_scaled = jax.tree_map(lambda x, y: x * jnp.abs(y) * self.prior_var ** 0.5, eps, params)

                pred_f, pred_jvp = jax.jvp(_pred_fn, (params,), (jax.tree_map(lambda x: x, eps_scaled),))
                preds_f = pred_f + pred_jvp
            else:
                _pred_fn = lambda params: self.model.apply(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=False,
                    mutable=False
                    )
                pred_f = _pred_fn(params)
                preds_f = pred_f

            if train:
                return preds_f, new_model_state
            else:
                return preds_f

        def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
            """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.
            Args:
                y: A pytree of numpy ndarray, vector y in the equation above.
            """
            gradient_norm = jnp.sqrt(sum(
                [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
            normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
            return normalized_gradient       

        # USED IN MEDFUSE
        def joint_fusion_inputs(inputs, _inputs_context):
            # Unpacking the tuple of size 5 for the train and context batches
            x_train, targets_ehr_train, img_train, targets_cxr_train, pairs = inputs
            x_context, targets_ehr_context, img_context, targets_cxr_context, pairs_context = _inputs_context

            # Concatenate each part of the fusion dataset
            x_joint = jnp.concatenate([x_train, x_context], axis=0)
            targets_ehr_joint = jnp.concatenate([targets_ehr_train, targets_ehr_context], axis=0)
            img_joint = jnp.concatenate([img_train, img_context], axis=0)
            targets_cxr_joint = jnp.concatenate([targets_cxr_train, targets_cxr_context], axis=0)
            pairs_joint = jnp.concatenate([pairs, pairs_context], axis=0)

            # We send a tuple of size 5 with all the concatenated elements
            inputs_joint = (x_joint, targets_ehr_joint, img_joint, targets_cxr_joint, pairs_joint)

            return inputs_joint
        
        # USED IN MEDFUSE
        def calculate_forward_pass(params, params_logvar, rng_key, batch_stats, inputs, _inputs_context, train):
            if self.linearize and self.pred_fn is None:
                self.pred_fn = f_lin
            else:
                self.pred_fn = self.model.apply

            preds_f_llk_list = []
            preds_f_reg_list = []
            params_samples = []
            for _ in range(self.mc_samples_llk):
                rng_key, _ = jax.random.split(rng_key)

                if self.objective_hparams["stochastic"]:
                    params = sample_parameters(params, params_logvar, self.stochastic, rng_key)
                    params_samples.append(params)

                if (
                    self.objective_hparams["forward_points"] == "train" and
                    self.objective_hparams["reg_points"] == "train"
                    ):
                    inputs_forward = inputs  # inputs used to update batch stats
                    inputs_reg = inputs_forward
                    # pdb.set_trace()
                    # a forward pass on the training points for the log-likelihood and regularization terms (batch stats are updated)
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs_forward,
                        train=train,
                        mutable=['batch_stats'] if train else False,
                        rngs={'dropout': rng_key}
                        )
                    preds_f_llk, new_model_state = out if train else (out, None)

                    preds_f_reg = preds_f_llk

                elif (
                    self.objective_hparams["forward_points"] == "train" and
                    self.objective_hparams["reg_points"] != "train"
                    ):
                    inputs_forward = inputs  # inputs used to update batch stats

                    # a forward pass on the training points for the log-likelihood term (batch stats are updated)
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs_forward,
                        train=train,
                        mutable=['batch_stats'] if train else False,
                        rngs={'dropout': rng_key}
                        )
                    preds_f_llk, new_model_state = out if train else (out, None)

                    if self.objective_hparams["reg_points"] == "context":
                        inputs_reg = _inputs_context
                        # a forward pass on the context points for the regularization term (batch stats are not updated)
                        out = self.pred_fn(
                            {'params': params, 'batch_stats': batch_stats},
                            inputs_reg,
                            train=train,
                            mutable=['batch_stats'] if train else False,
                            rngs={'dropout': rng_key}
                            )
                        _preds_f_reg, _ = out if train else (out, None)
                        preds_f_reg = jnp.concatenate([preds_f_llk, _preds_f_reg], axis=0)

                    elif self.objective_hparams["reg_points"] == "joint":
                        if self.dataset == "mimic-fusion":
                            inputs_reg = joint_fusion_inputs(inputs, _inputs_context)
                        else:
                            inputs_reg = jnp.concatenate([inputs, _inputs_context], axis=0)

                        # a forward pass on the joint points (context + train) for the regularization term (batch stats are not updated)
                        out = self.pred_fn(
                            {'params': params, 'batch_stats': batch_stats},
                            inputs_reg,
                            train=train,
                            mutable=['batch_stats'] if train else False,
                            rngs={'dropout': rng_key}
                            )
                        preds_f_reg, _ = out if train else (out, None)

                    else:
                        raise ValueError("Unknown forward_points/reg_points/context_points combination.")

                elif self.objective_hparams["forward_points"] == "joint":
                    if self.dataset == "mimic-fusion":
                        inputs_forward = joint_fusion_inputs(inputs, _inputs_context)
                    else:
                        inputs_forward = jnp.concatenate([inputs, _inputs_context], axis=0)  # inputs used to update batch stats

                    # a forward pass on both training and context points (batch stats are updated)
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs_forward,
                        train=train,
                        mutable=['batch_stats'] if train else False,
                        rngs={'dropout': rng_key}
                        )
                    preds_f_joint, new_model_state = out if train else (out, None)

                    if self.dataset == "mimic-fusion":
                        preds_f_llk = preds_f_joint[:inputs[0].shape[0]] # TODO We split preds up until the first mini batch?
                    else:
                        preds_f_llk = preds_f_joint[:inputs.shape[0]]

                    if self.objective_hparams["reg_points"] == "context":
                        inputs_reg = _inputs_context
                        if self.dataset == "mimic-fusion":
                            preds_f_reg = preds_f_joint[-inputs_reg[0].shape[0]:] # TODO We split preds up until the first mini batch?
                        else:
                            preds_f_reg = preds_f_joint[-inputs_reg.shape[0]:]

                    elif self.objective_hparams["reg_points"] == "joint":
                        if self.dataset == "mimic-fusion":
                            inputs_reg = joint_fusion_inputs(inputs, _inputs_context)
                        else:
                            inputs_reg = jnp.concatenate([inputs, _inputs_context], axis=0)
                        
                        preds_f_reg = preds_f_joint

                    else:
                        raise ValueError("Unknown forward_points/reg_points/context_points combination.")

                else:
                    raise ValueError("Unknown forward_points/reg_points/context_points combination.")
                    
                preds_f_llk_list.append(preds_f_llk)
                preds_f_reg_list.append(preds_f_reg)

            preds_f_llk = jnp.stack(preds_f_llk_list, axis=0)
            preds_f_reg = jnp.stack(preds_f_reg_list, axis=0)

            return preds_f_llk, preds_f_reg, params_samples, new_model_state, inputs_reg

        # USED IN MEDFUSE
        def calculate_loss(params, params_logvar, epsilon, epsilon_logvar, rng_key, batch_stats, batch, batch_context, train):
            if self.dataset == "mimic-fusion":
                # For Fusion in either task we use the EHR labels
                _, targets, _, _, _ = batch[0]
                inputs = batch[0]
                _inputs_context = batch_context[0]
            elif (self.dataset == "mimic-cxr" or self.dataset == "mimic-ehr") and (self.mimic_task == "in-hospital-mortality" or self.mimic_task == "phenotyping"):
                # For Uni-CXR we use the EHR labels
                if self.dataset == "mimic-cxr":
                    _, targets, inputs, _, _ = batch[0]
                elif self.dataset == "mimic-ehr":
                    inputs, targets, _, _, _ = batch[0]
                _inputs_context = batch_context[0]
            else:
                inputs, targets = batch
                _inputs_context, _ = batch_context
                
            preds_f_llk, preds_f_reg, params_samples, new_model_state, inputs_reg = calculate_forward_pass(params, params_logvar, rng_key, batch_stats, inputs, _inputs_context, train)

            if self.reg_type == "parameter_kl":
                assert self.objective_hparams["stochastic"] == True
                reg = calculate_parameter_kl(params, params_logvar)
            elif self.reg_type == "function_kl":
                reg = calculate_function_kl(params, params_logvar, inputs_reg, batch_stats, rng_key)
            elif self.reg_type == "function_prior":
                reg = calculate_function_prior_density(preds_f_reg, params, inputs_reg, batch_stats, rng_key, self.prior_var)
            elif self.reg_type == "entropic_prior":
                reg = calculate_entropic_prior(preds_f_reg)
            elif self.reg_type == "doubly_entropic_prior":
                reg = calculate_entropic_prior(preds_f_reg)
                reg += calculate_function_prior_density(preds_f_reg, params, inputs_reg, batch_stats, rng_key)
            elif self.reg_type == "empirical_gaussian_prior":
                reg = calculate_function_prior_density(preds_f_reg, params, inputs_reg, batch_stats, rng_key, self.prior_likelihood_scale)
                reg += calculate_parameter_norm(params, self.prior_var)
            elif self.reg_type == "empirical_categorical_prior":
                reg = calculate_empirical_categorical_prior(preds_f_reg, self.prediction_type)
                reg += calculate_parameter_norm(params, self.prior_var)
            elif self.reg_type == "empirical_gaussian_prior_kl":
                reg = calculate_empirical_gaussian_prior_kl(params_samples, params, params_logvar, preds_f_reg, inputs_reg, batch_stats, self.prior_likelihood_scale, self.prior_likelihood_f_scale, self.prior_likelihood_cov_scale, self.prior_likelihood_cov_diag, rng_key)
            elif self.reg_type == "empirical_categorical_prior_kl":
                reg = calculate_empirical_categorical_prior_kl(params_samples, params_logvar, preds_f_reg, self.prior_var, self.prediction_type)
            elif self.reg_type == "function_norm":
                reg = calculate_function_norm(preds_f_reg, inputs_reg, batch_stats, self.prior_var)
            elif self.reg_type == "parameter_norm":
                reg = calculate_parameter_norm(params, self.prior_var)
            else:
                raise ValueError("Unknown regularization type.")

            scale = 1 / self.n_batches_train  # 1 / (number of mini-batches)
            reg = scale * reg

            if self.output_var:
                likelihood_scale = preds_f_llk[:, :, self.num_classes:]
                preds_f_llk = preds_f_llk[:, :, :self.num_classes]
            else:
                likelihood_scale = self.likelihood_scale

            if self.prediction_type == "classification":
                nll = categorical_nll_with_softmax(jax.nn.softmax((1 / likelihood_scale) * preds_f_llk, -1), targets).mean(0).sum()  # likelihood_scale = temperature
                acc = 100 * (preds_f_llk.argmax(axis=-1) == targets).mean()
            elif self.prediction_type == "regression":
                nll = gaussian_nll(preds_f_llk, targets, likelihood_scale).mean(0).sum()  # likelihood_scale = likelihood variance
                acc = jnp.mean(jnp.mean(jnp.square(jnp.mean(preds_f_llk, axis=0) - targets), axis=-1), axis=-1)
            elif self.prediction_type == "mimic":
                logits = preds_f_llk
                num_labels = targets.shape[-1]
                preds = jnp.mean(jax.nn.sigmoid(logits), axis=0)
                # This sum is for the CLINICAL CONDITION taks were we have 25 losses
                nll = jnp.sum(mimic_ce_loss_sigmoid(preds, targets)) / num_labels

                if num_labels == 25: # Accuracy for CLINICAL CONDITION
                    acc = 0
                    for label in range(num_labels):
                        preds_b = preds[:, label]
                        targets_b = targets[:, label]
                        preds_b = (preds_b > 0.5).astype(float) 
                        label_equals = (preds_b == targets_b)
                        label_accuracy = 100 * jnp.array(jnp.mean(label_equals))
                        acc += label_accuracy
                    acc /= num_labels
                else: # Accuracy for MORTALITY
                    acc_preds = (preds > 0.5).astype(float)
                    label_equals = (acc_preds == targets)
                    label_accuracy = 100 * jnp.array(jnp.mean(label_equals))
                    acc = label_accuracy

                if self.gap_regularizer:
                    reg_gap, _ = calculate_sharpness_prior_gap(params, batch_stats, inputs, targets, rng_key, train=True)
                    
                    reg_gap = scale * reg_gap
            else:
                raise ValueError("Unknown prediction type.")

            if self.gap_regularizer:
                loss = (nll + self.reg_scale * reg + self.reg_scale_gap * reg_gap) / self.batch_size  # per data point loss
            else:
                loss = (nll + self.reg_scale * reg) / self.batch_size  # per data point loss

            if self.debug_print_updated:
                jax.debug.print("nll: {}", nll)
                jax.debug.print("reg: {}", reg)
                jax.debug.print("loss: {}", loss)
                jax.debug.print("acc: {}", acc)

            if self.prediction_type == "mimic":
                return loss, (acc, new_model_state, preds, targets)
            else:
                return loss, (acc, new_model_state)

        @partial(jit, static_argnums=(5,))
        def pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key, feature=False):
            params = sample_parameters(params, params_logvar, self.stochastic, rng_key)

            if feature:
                preds_f, feature = self.pred_fn(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=False,
                    mutable=False,
                    feature=feature,
                    rngs={'dropout': rng_key}
                    )
                return preds_f, feature        
            else:
                preds_f = self.pred_fn(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=False,
                    mutable=False,
                    rngs={'dropout': rng_key}
                    )
                return preds_f

        def evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, split):
            print(f"\nObtaining predictions for {split} dataset...")
            if split == "test" or split == "val" or split == "train":
                _logits_test = []
                _targets_test = []

                if self.prediction_type == "mimic":
                    if split == "val":
                        loader = enumerate(val_loader)
                    elif split == "test":
                        loader = enumerate(test_loader)
                    elif split == "train":
                        # --------------------------------------------------------------------
                        # Load train set to build the CONTEXT DATASET**
                        loader = enumerate(train_loader)
                        # loader = enumerate(test_loader)
                        # Create lists to save the whole dataset to build CONTEXT SET
                        ehr_inputs_train = []
                        ehr_targets_train = []
                        cxr_inputs_train = []
                        cxr_targets_train = []
                        ehr_cxr_pairs_train = []
                        # --------------------------------------------------------------------
                    else:
                        raise ValueError("The selected type of prediction to compute is unknown: {split}")
                else:
                    loader = enumerate(zip(test_loader, context_loader))
                    
                # for i, (batch, batch_2) in enumerate(zip(test_loader, context_loader)):
                for i, batch in loader:
                    if self.prediction_type == "mimic":
                        if (self.dataset == "mimic-cxr" or self.dataset == "mimic-ehr") and (self.mimic_task == "phenotyping" or self.mimic_task == "in-hospital-mortality"):
                            if self.dataset == "mimic-cxr": 
                                # Uni-CXR we extract the CXR inputs and EHR targets from the Fusion data tuple
                                _, targets, inputs, _, _ = batch[0]
                            elif self.dataset == "mimic-ehr":
                                 # Uni-EHR we extract the EHR inputs and EHR targets from the Fusion data tuple
                                inputs, targets, _, _, _ = batch[0]
                        elif self.model_name == "Fusion":
                            # Fusion-any-task we extract the EHR targets from the Fusion data tuple
                            # send the whole batch as the inputs, we unpack within the Fusion model
                            _, targets, _, _, _ = batch[0]
                            inputs = batch[0]
                        else:
                            # Uni-CXR-radiology or Uni-EHR, we extract the inputs and targets from the dataloader tuple
                            inputs, targets = batch
                        _targets = targets
                    else:
                        batch, batch_2 = batch
                        inputs_test, _targets = batch
                        inputs_context, _ = batch_2
                        n_context_points = inputs_context.shape[0]
                        inputs = jnp.concatenate([inputs_test, inputs_context], axis=0)
                    _logits_test_list = []

                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        if self.prediction_type == "mimic":
                            pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        else:
                            _pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                            pred = _pred[:_pred.shape[0] - n_context_points]
                        
                        _logits_test_list.append(pred)
                    
                    # --------------------------------------------------------------------
                    # Code used to create our CONTEXT DATASET**
                    if split == "train":
                        # Unpacking the tuple of size 5 for the train and context batches
                        x_train, targets_ehr_train, img_train, targets_cxr_train, ehr_cxr_pairs = inputs

                        ehr_inputs_train.append(x_train)
                        ehr_targets_train.append(targets_ehr_train)
                        cxr_inputs_train.append(img_train)
                        cxr_targets_train.append(targets_cxr_train)
                        ehr_cxr_pairs_train.append(ehr_cxr_pairs)
                    # --------------------------------------------------------------------

                    _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                    _targets_test.append(_targets)
                print("Done.")

                if split == "val":
                    logits_test = jnp.concatenate(_logits_test, axis=1)[:, :val_dataset_size, :]
                elif split == "train":
                    # Code used to create our CONTEXT DATASET**
                    logits_test = jnp.concatenate(_logits_test, axis=1)[:, :training_dataset_size, :]
                else:
                    logits_test = jnp.concatenate(_logits_test, axis=1)[:, :test_dataset_size, :]

                targets_test = jnp.concatenate(_targets_test, axis=0)
                ret = [logits_test, targets_test]

                # --------------------------------------------------------------------
                # Code used to create our CONTEXT DATASET**
                if split == "train":
                    all_TRAIN_ehr_inputs = jnp.concatenate(ehr_inputs_train, axis=0)
                    all_TRAIN_ehr_targets = jnp.concatenate(ehr_targets_train, axis=0)
                    all_TRAIN_cxr_inputs = jnp.concatenate(cxr_inputs_train, axis=0)
                    all_TRAIN_cxr_targets = jnp.concatenate(cxr_targets_train, axis=0)
                    all_TRAIN_ehr_cxr_pairs = jnp.concatenate(ehr_cxr_pairs_train, axis=0)

                    # ehr_targets and targets should be the same
                    np.savez(f"train_points_for_context_{self.mimic_task}_original_splits.npz", 
                    # np.savez(f"test_patient_datapoints_{self.mimic_task}_seed_{self.seed}.npz", 
                             ehr_inputs=all_TRAIN_ehr_inputs, 
                             ehr_targets=all_TRAIN_ehr_targets, 
                             cxr_inputs=all_TRAIN_cxr_inputs, 
                             cxr_targets=all_TRAIN_cxr_targets, 
                             ehr_cxr_pairs=all_TRAIN_ehr_cxr_pairs, 
                             preds=logits_test, 
                             targets=targets_test)
                # --------------------------------------------------------------------
            return ret
        
        #IMPORTANT PIECE OF CODE MEDFUSE
        def eval_patient_groups(params, params_logvar, rng_key, batch_stats, n_batches_eval):

            age_groups = {
                "AUROC":[],
                "AUPRC":[],
                "SEL_AUROC":[],
                "SEL_AUPRC":[],
                "GROUPS":["18-45", "45-60", "60-115"]
            }

            gender_groups = {
                "AUROC":[],
                "AUPRC":[],
                "SEL_AUROC":[],
                "SEL_AUPRC":[],
                "GROUPS":["M", "F"]
            }

            print("Obtaining model performance on AGE groups...")
            
            patients = pd.read_csv(MIMIC_PATIENT_METADATA_FILE)
            subject_ids = np.array([int(item.split('_')[0]) for item in test_loader.dataset.ehr_files_paired])

            selected = patients[patients.subject_id.isin(subject_ids)]
            start, step = 18, 15
            copy_ehr = np.copy(test_loader.dataset.ehr_files_paired)
            copy_cxr = np.copy(test_loader.dataset.cxr_files_paired)

            for i in [30, 45, 100]:
                subjects = selected.loc[((selected.anchor_age >= start) & (selected.anchor_age < i + step))].subject_id.values
                indexes = [jj for (jj, subject) in enumerate(subject_ids) if  subject in subjects]
                
                test_loader.dataset.ehr_files_paired = copy_ehr[indexes]
                test_loader.dataset.cxr_files_paired = copy_cxr[indexes]

                logits_test, targets_test = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "test")
                preds_test = jnp.mean(jax.nn.sigmoid(logits_test), axis=0)

                # AUROC
                auroc = roc_auc_score(targets_test, preds_test, average="weighted")

                # AUPRC
                auprc = average_precision_score(targets_test, preds_test, average="weighted")

                # Selective AUROC
                sel_auroc_ep1, _ = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUROC", model_name=f"{self.inference_data_model_name}", save_plot_data=self.save_plot_data)

                # Selective AUPRC
                sel_auprc_ep1, _ = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUPRC", model_name=f"{self.inference_data_model_name}", save_plot_data=self.save_plot_data)

                age_groups["AUROC"].append(auroc)
                age_groups["AUPRC"].append(auprc)
                age_groups["SEL_AUROC"].append(sel_auroc_ep1)
                age_groups["SEL_AUPRC"].append(sel_auprc_ep1)
                start = i + step

            print("Obtaining model performance on GENDER groups...")
            for gender in ["F", "M"]:
                subjects = selected.loc[((selected.anchor_age >= 18) & (selected.gender == gender))].subject_id.values
                indexes = [jj for (jj, subject) in enumerate(subject_ids) if  subject in subjects]
                
                test_loader.dataset.ehr_files_paired = copy_ehr[indexes]
                test_loader.dataset.cxr_files_paired = copy_cxr[indexes]

                logits_test, targets_test = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "test")
                preds_test = jnp.mean(jax.nn.sigmoid(logits_test), axis=0)

                # AUROC
                auroc = roc_auc_score(targets_test, preds_test, average="weighted")

                # AUPRC
                auprc = average_precision_score(targets_test, preds_test, average="weighted")

                # Selective AUROC
                sel_auroc_ep1, _ = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUROC", model_name=f"{self.inference_data_model_name}", save_plot_data=self.save_plot_data)

                # Selective AUPRC
                sel_auprc_ep1, _ = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUPRC", model_name=f"{self.inference_data_model_name}", save_plot_data=self.save_plot_data)
                
                gender_groups["AUROC"].append(auroc)
                gender_groups["AUPRC"].append(auprc)
                gender_groups["SEL_AUROC"].append(sel_auroc_ep1)
                gender_groups["SEL_AUPRC"].append(sel_auprc_ep1)
            
            print(f"AGE METRICS: {age_groups['GROUPS']}")
            print("AUROC")
            for element in age_groups["AUROC"]:
                print(element)
            print()

            print("AUPRC")
            for element in age_groups["AUPRC"]:
                print(element)
            print()

            print("SEL AUROC")
            for element in age_groups["SEL_AUROC"]:
                print(element)
            print()

            print("SEL AUPRC")
            for element in age_groups["SEL_AUPRC"]:
                print(element)
         
            print()
            print(f"GENDER METRICS: {gender_groups['GROUPS']}")
            
            print("AUROC")
            for element in gender_groups["AUROC"]:
                print(element)
            
            print()
            print("AUPRC")
            for element in gender_groups["AUPRC"]:
                print(element)
            
            print()
            print("SEL AUROC")
            for element in gender_groups["SEL_AUROC"]:
                print(element)
            
            print()
            print("SEL AUPRC")
            for element in gender_groups["SEL_AUPRC"]:
                print(element)

        #IMPORTANT PIECE OF CODE MEDFUSE
        def calculate_metrics(params, params_logvar, rng_key, batch_stats, n_batches_eval, final_eval, build_context_set):
            if self.prediction_type == "mimic":
                split = "val"
                if final_eval and build_context_set:
                    raise ValueError("Incorrect combination of parameters 'final_eval' and 'build_context_set'. Can't run final evaluation and extract data from TRAIN set.")
                elif final_eval:
                    split = "test"  # This means we are running calculate_metrics from the final eval mode, not during the training loop
                elif build_context_set:
                    split = "train" # We are extracting the data from the train set to build the context set
                logits_test, targets_test = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, split)
                
                preds_test = jnp.mean(jax.nn.sigmoid(logits_test), axis=0)
                num_labels = preds_test.shape[-1]
                
                if self.save_plot_data:
                    np.savez(f"data/MedFuse/inference_data/{self.inference_data_model_name}.npz", 
                                model_name=self.model_name, 
                                # Preds
                                preds=preds_test,
                                # Targets
                                targets=targets_test,
                                )
                    print(f"Saved preds and targets of loaded model on TEST set. Done.")

                if self.prediction_type == "mimic" and not build_context_set and not self.save_plot_data:
                    # We run this when we want to obtain the final CIs for all metrics.
                    # TODO: Revisit the code
                    if self.confidence_intervals and split == "test":
                        # AUROC
                        # auroc, auroc_per_label, auroc_ci = medfuse_ci.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="AUROC")
                        auroc, auroc_ci = medfuse_ci_2.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="AUROC")
                        # auroc_label = roc_auc_score(targets_test, preds_test, average=None)
                        
                        # print(f"AUROC {auroc}\n{auroc_per_label}\n{auroc_ci}\n")
                        print(f"AUROC {auroc}\n{auroc_ci}\n")
                        
                        # AUPRC
                        # auprc, auprc_per_label, auprc_ci = medfuse_ci.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="AUPRC")                    
                        auprc, auprc_ci = medfuse_ci_2.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="AUPRC")                    
                        # auprc_label = average_precision_score(targets_test, preds_test, average=None)
                        # print(f"AUPRC {auprc}\n{auprc_per_label}\n{auprc_ci}\n")
                        print(f"AUPRC {auprc}\n{auprc_ci}\n")

                        # Selective AUROC
                        # sel_auroc_ep1, sel_auroc_ep1_per_label, sel_auroc_ep1_ci = medfuse_ci.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="SEL_AUROC_1")  
                        sel_auroc_ep1, sel_auroc_ep1_ci = medfuse_ci_2.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="SEL_AUROC_1")   

                        # print(f"SEL AUROC {sel_auroc_ep1}\n{sel_auroc_ep1_per_label}\n{sel_auroc_ep1_ci}\n")
                        print(f"SEL AUROC {sel_auroc_ep1}\n{sel_auroc_ep1_ci}\n")

                        # Selective AUPRC
                        # sel_auprc_ep1, sel_auprc_ep1_per_label, sel_auprc_ep1_ci = medfuse_ci.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="SEL_AUPRC_1")  
                        sel_auprc_ep1, sel_auprc_ep1_ci = medfuse_ci_2.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="SEL_AUPRC_1")  
                        # print(f"SEL AUPRC {sel_auprc_ep1}\n{sel_auprc_ep1_per_label}\n{sel_auprc_ep1_ci}\n")
                        print(f"SEL AUPRC {sel_auprc_ep1}\n{sel_auprc_ep1_ci}\n")

                        # ECE
                        # ece, ece_per_label, ece_ci = medfuse_ci.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="ECE")
                        ece, ece_ci = medfuse_ci_2.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="ECE")
                        # print(f"ECE {ece}\n{ece_per_label}\n{ece_ci}\n")
                        print(f"ECE {ece}\n{ece_ci}\n")

                        # NLL
                        # nll, nll_ci = medfuse_ci.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="NLL")
                        nll, nll_ci = medfuse_ci_2.compute_metric_and_ci(y_true=targets_test, predictions=preds_test, selected_metric="NLL")
                        print(f"NLL {nll} {nll_ci}")

                        # NLL median
                        nll_median = jnp.median(mimic_ce_loss_sigmoid(preds_test, targets_test))
                       
                        # NLL sum
                        nll_sum = jnp.sum(mimic_ce_loss_sigmoid(preds_test, targets_test))

                        # print("Confidence intervals")
                        # print(f"AUROC {auroc:.3f} {auroc_ci}\nAUPRC {auprc:.3f} {auprc_ci}")
                        # print(f"\nSEL_AUROC {sel_auroc_ep1:.3f} {sel_auroc_ep1_ci}\nSEL_AUPRC {sel_auprc_ep1:.3f} {sel_auprc_ep1_ci}")
                        # print(f"\nNLL {nll:.3f} {nll_ci}\nECE {ece:.3f} {ece_ci}")
                        # print(f"\nNLL: SUM {nll_sum} | NLL_median {nll_median}")

                        # print(f"{auroc:.3f} ({auroc_ci[0]:.3f}, {auroc_ci[1]:.3f} )\n{auprc:.3f} {auprc_ci[0]:.3f}, {auprc_ci[1]:.3f})")
                        # print(f"\n{sel_auroc_ep1/100:.3f} ({sel_auroc_ep1_ci[0]/100:.3f}, {sel_auroc_ep1_ci[1]/100:.3f})\n{sel_auprc_ep1/100:.3f} ({sel_auprc_ep1_ci[0]/100:.3f}, {sel_auprc_ep1_ci[1]/100:.3f})")
                        # print(f"\n{nll:.3f} ({nll_ci[0]:.3f}, {nll_ci[1]:.3f})\n{ece:.3f} ({ece_ci[0]:.3f}, {ece_ci[1]:.3f})")
                        
                    else:
                        # Is performance affected by "average" flag??
                        # AUROC
                        auroc = roc_auc_score(targets_test, preds_test, average="weighted")
                        # auroc_labels = roc_auc_score(targets_test, preds_test, average=None)

                        # AUPRC
                        auprc = average_precision_score(targets_test, preds_test, average="weighted")
                        # auprc_labels = average_precision_score(targets_test, preds_test, average=None)

                        # Selective AUROC
                        selective_auroc, _ = selective_prediction_mimic(targets_test, preds_test, eval_metric="AUROC")
                        # sel_auroc_ep1, sel_auroc_ep1_labels = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUROC", model_name=f"{self.inference_data_model_name}", save_plot_data=self.save_plot_data)

                        # Selective AUPRC
                        selective_auprc, _ = selective_prediction_mimic(targets_test, preds_test, eval_metric="AUPRC")
                        # sel_auprc_ep1, sel_auprc_ep1_labels = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUPRC", model_name=f"{self.inference_data_model_name}", save_plot_data=self.save_plot_data)
                        
                        # ECE
                        ece_split_dict = {}
                        total_ece_split = 0

                        for label in range(num_labels):
                            # We extract the preds/targets per label
                            preds_test_label = preds_test[:, label]
                            targets_test_label = targets_test[:,label]

                            ece_preds_test_label = jnp.column_stack((1-preds_test_label, preds_test_label))
                            ece_split_label, _ = calibration(targets_test_label, ece_preds_test_label)

                            ece_split_dict[f'label_{label+1}'] = ece_split_label
                            total_ece_split += ece_split_label

                        ece_split_dict['total_ece_split'] = total_ece_split/num_labels
                        ece = ece_split_dict['total_ece_split']

                        # NLL
                        nll = jnp.mean(mimic_ce_loss_sigmoid(preds_test, targets_test))
                        nll_sum = jnp.sum(mimic_ce_loss_sigmoid(preds_test, targets_test))
                        nll_median = jnp.median(mimic_ce_loss_sigmoid(preds_test, targets_test))

                        # AGE & GENDER ANALYSIS
                        if self.eval_patient_groups:
                            eval_patient_groups(params, params_logvar, rng_key, batch_stats, n_batches_eval)

                    self.logger[f"auroc_{split}"].append(auroc)
                    self.logger[f"auroc_selective_{split}"].append(selective_auroc)

                    self.logger[f"auprc_{split}"].append(auprc)
                    self.logger[f"auprc_selective_{split}"].append(selective_auprc)

                    auroc_auprc_avg = (auroc + auprc)/2
                    self.logger[f"auroc_auprc_{split}"].append(auroc_auprc_avg)

                    self.logger[f"nll_{split}"].append(nll)
                    self.logger[f"nll_median_{split}"].append(nll_median)
                    self.logger[f"nll_sum_{split}"].append(nll_sum)
                    self.logger[f"loss_{split}"].append(nll)
                    self.logger[f"ece_{split}"].append(ece)

                    if self.confidence_intervals and split == "test":
                        self.logger[f"auprc_ci_{split}"].append(auprc_ci)
                        self.logger[f"auroc_ci_{split}"].append(auroc_ci)
                
        @partial(jit, static_argnums=(4,))
        def train_step(state, batch, batch_context, rng_key, debug_print):
            self.debug_print_updated = debug_print

            params = state.params["params"]
            params_logvar = state.params["params_logvar"]

            if self.rho_sam != 0 and self.prediction_type == "classification" and self.reg_type == "parameter_norm":

                inputs, targets = batch
                _inputs_context, _ = batch_context

                loss_fn_sam = lambda params, params_logvar: categorical_nll_with_softmax(
                    jax.nn.softmax((1 / self.likelihood_scale) * calculate_forward_pass(params, params_logvar, rng_key, state.batch_stats, inputs, _inputs_context, train=True)[0], -1), targets
                    ).mean(0).sum()

                _, _grads = jax.value_and_grad(loss_fn_sam, argnums=(0,1,))(params, params_logvar)
                grads, grads_logvar = _grads[0], _grads[1]
                epsilon, epsilon_logvar = dual_vector(grads), dual_vector(grads_logvar)
            else:
                epsilon, epsilon_logvar = 0, 0

            loss_fn = lambda params, params_logvar: calculate_loss(params, params_logvar, epsilon, epsilon_logvar, rng_key, state.batch_stats, batch, batch_context, train=True)
            # Get loss, gradients for loss, and other outputs of loss function
            ret, _grads = jax.value_and_grad(loss_fn, argnums=(0,1,), has_aux=True)(params, params_logvar)
            grads, grads_logvar = _grads[0], jax.tree_map(lambda x: self.learning_rate_scale_logvar * x, _grads[1])
            
            if self.prediction_type == "mimic":
                loss, acc, new_model_state, preds, targets = ret[0], *ret[1]
                # Update parameters and batch statistics
                state = state.apply_gradients(grads=freeze({"params": grads, "params_logvar": grads_logvar}), batch_stats=new_model_state['batch_stats'])
            
                return state, loss, acc, preds, targets
            else:
                loss, acc, new_model_state = ret[0], *ret[1]
                # Update parameters and batch statistics
                state = state.apply_gradients(grads=freeze({"params": grads, "params_logvar": grads_logvar}), batch_stats=new_model_state['batch_stats'])
                return state, loss, acc

        def eval_step(state, rng_key, n_batches_eval, final_eval, build_context_set):
            calculate_metrics(state.params["params"], state.params["params_logvar"], rng_key, state.batch_stats, n_batches_eval, final_eval, build_context_set)
    
        self.train_step = train_step
        self.evaluation_predictions = evaluation_predictions
        self.eval_step = eval_step
        self.pred_fn_jit = pred_fn_jit

    def init_model(self, exmp_inputs):
        print("Starting model initialization...")
        root_key = jax.random.PRNGKey(self.seed)
        params_rng, dropout_key  = random.split(root_key, num=2)
        init_rng = {'params': params_rng, 'dropout': dropout_key}

        variables = self.model.init(init_rng, exmp_inputs, train=True)
        variables_logvar = self.model.init(init_rng, exmp_inputs, train=True)

        # Initialize mean parameters using a pre-trained prior
        if self.det_model_fine_tuning and self.pretrained_prior and self.pretrained_prior_path != "":
            print(f" DET model fine-tuning: Initializing mean parameters with pretrained prior:\n  {self.pretrained_prior_path}.")
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.pretrained_prior_path, target=None)
            state_dict["params"]["fused_cls"] = variables["params"]["fused_cls"]
            state_dict = freeze(state_dict)
            init_params  = state_dict["params"]
        elif self.pretrained_prior and self.pretrained_prior_path != "":
            print(f" Initializing mean parameters with pretrained prior:\n {self.pretrained_prior_path}")
            state_dict = freeze(checkpoints.restore_checkpoint(ckpt_dir=self.pretrained_prior_path, target=None))
            init_params  = state_dict["params"]
        elif self.pretrained_prior and self.pretrained_prior_path == "":
            raise ValueError(f"You are trying to initialize the model's mean parameters but have not provided a correct pre-trained model. Model path selected: {self.pretrained_prior_path}")
        else:
            print(f" Initializing random params. No model loaded")
            init_params = variables["params"] # Random initialization

        if self.stochastic:
            init_params_logvar = jax.tree_map(lambda x: x + self.init_logvar, variables_logvar['params'])

            if self.prediction_type == "mimic":
                init_params_final_layer_logvar, init_params_feature_logvar = split_params(init_params_logvar, "dense")
            else:   
                init_params_feature_logvar, init_params_final_layer_logvar = split_params(init_params_logvar, "dense")

            init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 + self.init_logvar, init_params_final_layer_logvar)
            self.final_layer_key = [key for key in init_params_final_layer_logvar.keys()][-1]
            
            minval_weights = self.init_final_layer_weights_logvar
            maxval_weights = self.init_final_layer_weights_logvar + 0.1

            # Changed the key that is sent to jax.random.uniform in case we are running Fusion since init_rng was a dict and jax.random.uniform does not accept that
            init_params_final_layer_logvar[self.final_layer_key]["kernel"] = jnp.array(jax.random.uniform(key=params_rng, shape=init_params_final_layer_logvar[self.final_layer_key]["kernel"].shape, minval=minval_weights, maxval=maxval_weights, dtype=float))
            
            minval_bias = self.init_final_layer_bias_logvar
            maxval_bias = self.init_final_layer_bias_logvar + 0.1
            init_params_final_layer_logvar[self.final_layer_key]["bias"] = jnp.array(jax.random.uniform(key=params_rng, shape=init_params_final_layer_logvar[self.final_layer_key]["bias"].shape, minval=minval_bias, maxval=maxval_bias, dtype=float))
            
            # init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 - 15, init_params_final_layer_logvar)
            # init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 - 50, init_params_final_layer_logvar)
            init_params_logvar = merge_params(init_params_feature_logvar, init_params_final_layer_logvar)
        else:
            init_params_logvar = None

        if self.model_name == "LSTM":
            # This was required because of the way the LSTM works in Flax
            self.init_batch_stats = {}
        else:
            self.init_batch_stats = variables["batch_stats"]

        self.init_params = freeze({"params": init_params, "params_logvar": init_params_logvar})

        self.linearization_params = jax.tree_map(lambda x: x * 1.00001, jax.lax.stop_gradient(self.init_params))
        self.linearization_batch_stats = jax.tree_map(lambda x: x * 1.00001, jax.lax.stop_gradient(self.init_batch_stats))

        self.state = None
        print("Finished model initialization.\n\n")

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        if self.optimizer_name.lower() == "adam":
            opt_class = optax.adam
            self.optimizer_hparams.pop("momentum")
            self.optimizer_hparams.pop("weight_decay")
        elif self.optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
            self.optimizer_hparams.pop("momentum")
        elif self.optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
            self.optimizer_hparams.pop("b1")
            self.optimizer_hparams.pop("b2")
        else:
            assert False, f"Unknown optimizer {self.optimizer_name} selected."

        self.learning_rate_scale_logvar = self.optimizer_hparams.pop("learning_rate_scale_logvar")
        
        if alpha != 1:
            lr_schedule = optax.cosine_decay_schedule(
                init_value=self.optimizer_hparams.pop("lr"),
                decay_steps=num_steps_per_epoch*num_epochs,
                alpha=self.optimizer_hparams.pop("alpha"),
                exponent=self.optimizer_hparams.pop("exponent")
            )
        else:
            lr_schedule = optax.piecewise_constant_schedule(
                init_value=self.optimizer_hparams.pop("lr")
            )
            self.optimizer_hparams.pop("alpha")

        transf = []
        # transf = [optax.clip(1.0)]
        if (opt_class == optax.sgd or opt_class == optax.adamw) and "weight_decay" in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop("weight_decay")))

        optimizer = optax.chain(*transf, opt_class(lr_schedule, **self.optimizer_hparams))
        
        # lr = self.optimizer_hparams.pop("lr")
        # self.optimizer_hparams.pop("alpha")
        # optimizer = optax.chain(*transf, opt_class(learning_rate=lr, **self.optimizer_hparams))
        #-------------------------
    
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            tx=optimizer
            )
        
    def train_model(self, train_loader, context_loader, val_loader, rng_key, num_epochs=200):
        print(f"\nTraining model for {num_epochs} epochs:\n")
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval = 0.0
        best_val_auroc = 0.0
        best_val_sel_auroc = 0.0
        best_val_auprc = 0.0
        best_val_auroc_auprc = 0.0

        # Only runs  if batch stats prior is None
        if self.batch_stats_prior is None and self.batch_stats_init_epochs != 0:
            print(f"Calibrating batch normalization statistics for {self.batch_stats_init_epochs} epochs:\n")
            self.batch_stats_prior = self.state.batch_stats
            self.state_pretrain = self.state
            for _ in tqdm(range(self.batch_stats_init_epochs)):
                self.pretrain(train_loader, context_loader, rng_key=rng_key)
            self.state = self.state.replace(batch_stats=self.state_pretrain.batch_stats)
            self.batch_stats_prior = self.state_pretrain.batch_stats

        for epoch in tqdm(range(num_epochs), leave=False):
            epoch_idx = epoch + 1
            self.train_epoch(train_loader, context_loader, epoch=epoch_idx, rng_key=rng_key)
            
            if epoch_idx % self.log_frequency == 0:
                if self.dataset != "two-moons" and self.dataset != "snelson" and self.dataset != "oat1d" and "offline_rl" not in self.dataset:
                    self.eval_model(rng_key, self.n_batches_eval)
                    self.eval_model(rng_key, self.n_batches_eval, final_eval=True)
                    
                    if self.prediction_type == "mimic":
                        print("Saving model checkpoints...")

                        # SAVING ON BEST AUROC VALIDATION
                        if self.logger['auroc_val'][-1] >= best_val_auroc:
                            self.logger['auroc_val_best'].append(self.logger['auroc_val'][-1])
                            best_val_auroc = self.logger['auroc_val_best'][-1]
                            if self.save_model_on == "AUROC":
                                self.save_model(step=epoch_idx, best=True) # We save the model if its the best
                        
                        # We do not save models based on AUPRC
                        if self.logger['auprc_val'][-1] >= best_val_auprc:
                            self.logger['auprc_val_best'].append(self.logger['auprc_val'][-1])
                            best_val_auprc = self.logger['auprc_val_best'][-1]

                        # # SAVING ON BEST SEL AUROC VALIDATION
                        if self.logger['auroc_selective_val'][-1] >= best_val_sel_auroc:
                            self.logger['auroc_selective_val_best'].append(self.logger['auroc_selective_val'][-1])
                            best_val_sel_auroc = self.logger['auroc_selective_val_best'][-1]
                            if self.save_model_on == "SEL_AUROC":
                                self.save_model(step=epoch_idx, best=True) # We save the model if its the best
                            
                        # # SAVING ON BEST AUROC_AUPRC AVG
                        if self.logger['auroc_auprc_val'][-1] >= best_val_auroc_auprc:
                            self.logger['auroc_auprc_val_best'].append(self.logger['auroc_auprc_val'][-1])
                            best_val_auroc_auprc = self.logger['auroc_auprc_val_best'][-1]
                            if self.save_model_on == "AUROC_AUPRC":
                                self.save_model(step=epoch_idx, best=True) # We save the model if its the best

                        print(f" Model checkpoint for best {self.save_model_on} from BEST-MODEL-SAVED section, done.")
                        
                    else: # Any other dataset and/or architecture than FUSION/MIMIC
                        if self.logger['acc_test'][-1] >= best_eval:
                            self.logger['acc_test_best'].append(self.logger['acc_test'][-1])
                            self.save_model(step=epoch_idx, best=True)
                        else:
                            self.logger['acc_test_best'].append(self.logger['acc_test_best'][-1])
                        best_eval = self.logger['acc_test_best'][-1]
                    
                    self.logger['epoch'].append(epoch_idx) 

                    if self.save_to_wandb and epoch_idx < num_epochs:
                        self.wandb_logger.append({})
                        for item in self.logger.items():
                            try:
                                self.wandb_logger[-1][item[0]] = item[1][-1]
                            except:
                                pass
                        wandb.log(self.wandb_logger[-1])

                    if self.prediction_type == "mimic":
                        self.save_model(step=epoch_idx) # We save every model anyway
                        print(" Model checkpoint from ALL-MODELS-SAVED section, done.")

                        print(f"\n**EPOCH {epoch_idx} FINISHED.**")
                        print(f"\n VAL SCORES:")
                        print(f" AUROC {self.logger['auroc_val'][-1]:.3f} | AUPRC {self.logger['auprc_val'][-1]:.3f} | SEL AUROC: {self.logger['auroc_selective_val'][-1]:.2f} | SEL AUPRC: {self.logger['auprc_selective_val'][-1]:.2f} | ECE: {self.logger['ece_val'][-1]:.2f} | NLL: {self.logger['nll_val'][-1]:.3f}")
                        if self.logger["auroc_test"] != 0:
                            print(f"\n TEST SCORES:")
                            print(f" AUROC {self.logger['auroc_test'][-1]:.3f} | AUPRC {self.logger['auprc_test'][-1]:.3f} | SEL AUROC: {self.logger['auroc_selective_test'][-1]:.2f} | SEL AUPRC: {self.logger['auprc_selective_test'][-1]:.2f} | ECE: {self.logger['ece_test'][-1]:.2f} | NLL: {self.logger['nll_test'][-1]:.3f}")
                        print("\n-------------------------------------------------------")
                        print("             MODEL EVALUATION FINISHED")
                        print("-------------------------------------------------------")

    def pretrain(self, train_loader, context_loader, rng_key):
        data_loader = tqdm(zip(train_loader, context_loader), leave=False)
        for batch, batch_context in data_loader:
            self.state_pretrain, loss, acc = self.train_step(self.state_pretrain, batch, batch_context, rng_key)
            rng_key, _ = jax.random.split(rng_key)
            self.batch_stats_prior = self.state_pretrain.batch_stats

    def train_epoch(self, train_loader, context_loader, epoch, rng_key):

        metrics = defaultdict(list)
        data_loader = tqdm(enumerate(zip(train_loader, context_loader)), leave=False)
        train_acc = 0
        elapsed = 0

        epoch_preds = []
        epoch_targets = []

        for i, (batch, batch_context) in data_loader:
            # print(f"\nEpoch {epoch} Batch {i}   \n")
            if self.debug_print:
                if i % 1000 == 0:
                    debug_print = True
                    print(f"\nEpoch {epoch} Batch {i}   \n")
                else:
                    debug_print = False
            else:
                debug_print = False

            if self.prediction_type == "mimic":
                self.state, loss, acc, preds, targets = self.train_step(self.state, batch, batch_context, rng_key, debug_print)
            else:
                self.state, loss, acc = self.train_step(self.state, batch, batch_context, rng_key, debug_print)
            
            rng_key, _ = jax.random.split(rng_key)
            metrics["loss"].append(loss)
            metrics["acc"].append(acc)

            if (data_loader.format_dict["elapsed"] - elapsed) >= 0.5:  # Update every 5 seconds
                
                train_acc = np.stack(jax.device_get(metrics["acc"]))[-40:].mean()  # Average accuracy of last 40 batches
                train_loss = np.stack(jax.device_get(metrics["loss"]))[-40:].mean()  # Average accuracy of last 40 batches
                data_loader.set_postfix({'accuracy': train_acc, 'loss': train_loss})
                
                if save_to_wandb and self.prediction_type == "mimic":
                    wandb.log({'train_step_loss': train_loss})
                elapsed = data_loader.format_dict["elapsed"]

            if self.prediction_type == "mimic":
                epoch_preds.append(preds)
                epoch_targets.append(targets)

        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger[f"{key}_train"].append(avg_val)
        
        if self.prediction_type == "mimic":
            preds = jnp.concatenate(epoch_preds, axis=0)
            targets = jnp.concatenate(epoch_targets, axis=0)
            auroc = roc_auc_score(targets, preds)
            auprc = average_precision_score(targets, preds)

            if save_to_wandb:
                wandb.log({'train_epoch_auroc': auroc, 'train_epoch_auprc': auprc})

            print(f"\n\nTRAIN EPOCH: {epoch} FINISHED.")
            print(f"\n AUROC: {auroc}  | AUPRC {auprc}")
            print("-------------------------------------------------------")

    def eval_model(self, rng_key, n_batches_eval, final_eval=False, build_context_set=False):
        if not build_context_set:
            if final_eval:
                print("\n\n-------------------------------------------------------")
                print("            MODEL EVALUATION ON TEST SET")
                print("-------------------------------------------------------")
            else:
                print("\n\n-------------------------------------------------------")
                print("           MODEL EVALUATION ON VALIDATION SET")
                print("-------------------------------------------------------")
        else:
            print("\n\n-------------------------------------------------------")
            print("           EXTRACTING DATA FROM TRAIN SET FOR CONTEXT SET")
            print("-------------------------------------------------------")
        self.eval_step(self.state, rng_key, n_batches_eval, final_eval, build_context_set)

    def save_model(self, step=0, best=False):
        if best:
            checkpoints.save_checkpoint(
                ckpt_dir=f"{self.log_dir}_{best}",
                target={
                    'params': self.state.params["params"],
                    'params_logvar': self.state.params["params_logvar"],
                    'batch_stats': self.state.batch_stats,
                    'batch_stats_prior': self.batch_stats_prior
                },
                step=0,
                overwrite=True,
                prefix=f"{self.save_model_on}_checkpoint_best_"
                )
            if self.save_to_wandb:
                wandb.save(f'{self.log_dir}/{self.save_model_on}_checkpoint_best_0')
        else:
            checkpoints.save_checkpoint(
                ckpt_dir=f"{self.log_dir}_{step}",
                target={
                    'params': self.state.params["params"],
                    'params_logvar': self.state.params["params_logvar"],
                    'batch_stats': self.state.batch_stats,
                    'batch_stats_prior': self.batch_stats_prior
                },
                step=step,
                overwrite=True,
                prefix=f"checkpoint_"
                )
            if self.save_to_wandb:
                wandb.save(f'{self.log_dir}/checkpoint_{step}')

    def checkpoint_exists(self):
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f"{self.model_name}.ckpt"))

def trainer(*args, rng_key, **kwargs):
    trainer = TrainerModule(*args, **kwargs)
    del kwargs["exmp_inputs"]

    pprint(kwargs)
    if trainer.save_to_wandb:
        wandb.config = copy(kwargs)
        wand_run = wandb.init(
            project=trainer.wandb_project,
            name=trainer.run_name,
            entity=trainer.wandb_account,
            config=wandb.config,
        )
    train = not trainer.evaluate
    
    if "Pretrained" in kwargs["model_name"]:
        prior = True
    else:
        prior = False

    # Train model from scratch
    if train and not trainer.restore_checkpoint and not "Pretrained" in kwargs["model_name"] and not trainer.build_context_set and not trainer.eval_patient_groups: 
        print("TRAINING FROM SCRATCH") 
        trainer.train_model(train_loader, context_loader, val_loader, rng_key, num_epochs=trainer.num_epochs)

    # Test evaluation at the end of training and construction of CONTEXT set
    # In any training/loading scenario, we will always load the best model and run the test set
    if trainer.prediction_type == "mimic":
        if trainer.model_for_final_eval == "BEST":
            best_models = [f"{trainer.log_dir}_True/{trainer.save_model_on}_checkpoint_best_0"]
        elif trainer.model_for_final_eval == "LAST":
            best_models = [f"{trainer.log_dir}_{trainer.num_epochs}/checkpoint_{trainer.num_epochs}"]
        else:
            best_models = [trainer.model_for_final_eval]
        
        # This loop runs inference on different models if desired.
        if trainer.build_context_set:
            print(f"\nLoading model to extract TRAIN data for CONTEXT set construction.")
            # print(f"\nLoading model to extract TEST data for CONTEXT set construction. MODIF TODO")
        else:
            print(f"\nLoading {trainer.model_for_final_eval} model(s) for final evaluation on test set.")
            
        for best_model_path in best_models:
            #-----------------------------------------------------------------
            print(f"\nLoading: {best_model_path}")
            
            state_dict = freeze(checkpoints.restore_checkpoint(ckpt_dir=best_model_path, target=None))
            
            if stochastic:
                params_logvar = state_dict['params_logvar']
            else:
                params_logvar = None

            if trainer.linearize and trainer.pred_fn is None:
                NotImplementedError
            else:
                trainer.pred_fn = trainer.model.apply

            params = freeze({"params": state_dict['params'], "params_logvar": params_logvar, 'dropout': rng_key})

            trainer.state = TrainState.create(apply_fn=trainer.model.apply,
                                        params=params,
                                        batch_stats=state_dict['batch_stats'],
                                        tx=trainer.state.tx if trainer.state else optax.sgd(0.1)   # Default optimizer
                                        )
            print(f"\nFinished loading best model checkpoint:\n {best_model_path}")
            print("------------------------------------------------------------\n\n")
            #---------------------------------------------------------------
            if trainer.build_context_set:
                trainer.eval_model(rng_key, trainer.n_batches_eval_final, final_eval=False, build_context_set=True)
            elif trainer.save_plot_data:
                print(f"Saving preds and targets from loaded model on TEST set...")
                trainer.eval_model(rng_key, trainer.n_batches_eval_final, final_eval=True, build_context_set=False)
            else:
                if not train:
                    trainer.logger['acc_train'].append(0)
                    trainer.logger['acc_train'].append(0)
                    trainer.logger['loss_train'].append(0)
                trainer.logger['epoch'].append(trainer.num_epochs)

                trainer.eval_model(rng_key, trainer.n_batches_eval_final, final_eval=True)
                print(f"\nFINAL TEST SET SCORES:")
                print(f" AUROC {trainer.logger['auroc_test'][-1]:.3f} | AUPRC {trainer.logger['auprc_test'][-1]:.3f} | SEL AUROC EP1: {trainer.logger['auroc_selective_test'][-1]:.3f} | SEL AUPRC EP1: {trainer.logger['auprc_selective_test'][-1]:.3f} | ECE: {trainer.logger['ece_test'][-1]:.3f} | NLL: {trainer.logger['nll_test'][-1]:.3f}")
                print(f"\nFinished evaluation on best model checkpoint\n {best_model_path}")
                print("------------------------------------------------------------\n\n")

    else:
        trainer.load_model(stochastic=trainer.stochastic, pretrained_prior=trainer.pretrained_prior, restore_checkpoint=trainer.restore_checkpoint)
        trainer.logger['acc_train'].append(0)
        trainer.logger['acc_test_best'].append(0)
        trainer.logger['loss_train'].append(0)
        trainer.logger['epoch'].append(trainer.num_epochs)

        if trainer.dataset != "two-moons" and trainer.dataset != "snelson" and trainer.dataset != "oat1d" and "offline_rl" not in trainer.dataset:
            # val_acc = trainer.eval_model(val_loader, rng_key)
            trainer.eval_model(rng_key, trainer.n_batches_eval_final, final_eval=trainer.final_eval)
            # print(f"\nValidation Accuracy: {val_acc*100:.2f}")
            print(f"Train Accuracy: {trainer.logger['acc_train'][-1]:.2f}  |  Test Accuracy: {trainer.logger['acc_test'][-1]:.2f}  |  Selective Accuracy: {trainer.logger['acc_sel_test'][-1]:.2f}  |  Selective Accuracy Test+OOD: {trainer.logger['acc_sel_test_ood'][-1]:.2f}  |  NLL: {trainer.logger['nll_test'][-1]:.3f}  |  Test ECE: {trainer.logger['ece_test'][-1]:.2f}  |  OOD AUROC: {trainer.logger['ood_auroc_entropy'][-1]:.2f} / {trainer.logger['ood_auroc_aleatoric'][-1]:.2f} / {trainer.logger['ood_auroc_epistemic'][-1]:.2f}  |  Uncertainty Test: {trainer.logger['predictive_entropy_test'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_test'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_test'][-1]:.3f}  |  Uncertainty Context: {trainer.logger['predictive_entropy_context'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_context'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_context'][-1]:.3f}  |  Uncertainty OOD: {trainer.logger['predictive_entropy_ood'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_ood'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_ood'][-1]:.3f}")

    if not trainer.build_context_set:
        trainer.wandb_logger.append({})
        
        print("Empty logger variables")
        for item in trainer.logger.items():
            try:
                trainer.wandb_logger[-1][item[0]] = item[1][-1]
            except:
                print(f" {item}")

        print("\nMAIN FINAL SCORES")
        main_metrics_test = ["auroc_test", "auprc_test","auroc_selective_test", 
                        "auprc_selective_test", "ece_test"]
        
        main_metrics_val = ["auroc_val_best",
                            "auprc_val_best", "auroc_selective_val_best", "auprc_selective_val", "ece_val"]
        print("\nTEST")
        for metric in main_metrics_test:
            print(f"{trainer.wandb_logger[-1][metric]}")
        if train:
            print("\nVAL")
            for metric in main_metrics_val:
                print(f"{trainer.wandb_logger[-1][metric]}")

        print("\nALL METRICS")
        pprint(trainer.wandb_logger[-1])
        if trainer.save_to_wandb:
            wandb.log(trainer.wandb_logger[-1])
            time.sleep(10)
    return trainer, trainer.logger
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# GENERAL SETUP OF MODEL ARCHITECTURE
if 'CNN' in model_name:
    model_class = CNN
    num_blocks = None
    c_hidden = None
if 'MLP_Toy' in model_name:
    model_class = MLP_Toy
    num_blocks = None
    c_hidden = None
if 'MLP_OfflineRL' in model_name:
    model_class = MLP_OfflineRL
    num_blocks = None
    c_hidden = None
if 'ResNet9' in model_name:  # 272,896 parameters for FMNIST
    model_class = ResNetMod
    num_blocks = (3, 3, 3)
    c_hidden = (16, 32, 64)
if 'ResNet18' in model_name:  # 11,174,642 parameters for FMNIST
    model_class = ResNetMod
    num_blocks = (2, 2, 2, 2)
    c_hidden = (64, 128, 256, 512)
if 'ResNet34' in model_name:
    model_class = ResNetMod
    num_blocks = (3, 4, 6, 3)
    c_hidden = (64, 128, 256, 512)
if 'ResNet50' in model_name:
    model_class = ResNetMod
    num_blocks = None
    c_hidden = (64, 128, 256, 512)
if 'LSTM' in model_name:
    model_class = lstm_jax.LSTM
    num_blocks = None
    c_hidden = None
if 'Fusion' in model_name:
    model_class = fusion_jax.Fusion
    num_blocks = None
    c_hidden = None

block_class = ResNetBlock
# block_class = PreActResNetBlock
act_fn = nn.relu
# act_fn = nn.swish
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# GENERAL SETUP OF METHOD
if prior_precision == 0:
    prior_precision = 1 / prior_var
elif prior_var == 0:
    prior_var = 1 / prior_precision
else:
    raise ValueError("Only one of prior_precision and prior_var can be set.")

prior_mean = "Pretrained Mean" if "Pretrained" in model_name else prior_mean

if method == "fsmap":
    stochastic = False
if method == "psmap":
    stochastic = False
if method == "fsvi":
    stochastic = True
if method == "psvi":
    stochastic = True
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# STARTING TRAINING
print("FINAL DATA LENGTHS USED FOR TRAINING:\n")
print(f" DATASETS\n Training: {len(train_dataset)} | Validation: {len(validation_dataset)} | Testing: {len(test_dataset)} | Context: {len(context_dataset)}")
print()
print(f" DATA LOADERS\n Training: {len(train_loader)} | Validation: {len(val_loader)} | Testing: {len(test_loader)} | Context: {len(context_loader)}")
if len(train_loader) > len(context_loader):
    raise ValueError("The size of your train loader is larger than the context loader.")
print("------------------------------------------------------------------------")
print("------------------------------------------------------------------------")

if full_training:
    print("\nBEGINNING FULL TRAINING (TRAIN+VAL)...\n") # Training with train+val sets for final model after finetuning
else:
    print("\nBEGINNING TRAINING...\n")

model_trainer, model_results = trainer(
    model_name=model_name,
    model_class=model_class,
    model_hparams={
        "num_classes": num_classes,
        "c_hidden": c_hidden,
        "num_blocks": num_blocks,
        "act_fn": act_fn,
        "block_class": block_class,
        },
    optimizer_name=optimizer_name,
    optimizer_hparams={
                        "lr": learning_rate,
                        "b1": 0.9,
                        "b2": 0.9,
                        "learning_rate_scale_logvar": learning_rate_scale_logvar,
                        "momentum": momentum,
                        "alpha": alpha,
                        "exponent": exponent,
                        "weight_decay": weight_decay,
                        },
    objective_hparams={
                        "method": method,
                        "stochastic": stochastic,
                        "reg_type": reg_type,
                        "reg_scale": reg_scale,
                        "reg_scale_gap": reg_scale_gap,
                        "gap_regularizer": gap_regularizer,
                        "prior_mean": prior_mean,
                        "prior_var": prior_var,
                        "prior_likelihood_scale": prior_likelihood_scale,
                        "prior_likelihood_f_scale": prior_likelihood_f_scale,
                        "prior_likelihood_cov_scale": prior_likelihood_cov_scale,
                        "prior_likelihood_cov_diag": prior_likelihood_cov_diag,
                        "prior_likelihood_mean": prior_likelihood_mean,
                        "prior_likelihood_normalize_feature": prior_likelihood_normalize_feature,
                        "likelihood_scale": likelihood_scale,
                        "rho_sam": rho_sam,
                        "context_points": context_points,
                        "forward_points": forward_points,
                        "reg_points": reg_points,
                        "mc_samples_llk": mc_samples_llk,
                        "mc_samples_reg": mc_samples_reg,
                        "training_dataset_size": training_dataset_size,
                        "batch_size": batch_size,
                        "init_logvar": init_logvar,
                        "init_final_layer_weights_logvar": init_final_layer_weights_logvar,
                        "init_final_layer_bias_logvar": init_final_layer_bias_logvar,
                        "prior_feature_logvar": prior_feature_logvar,
                        "pretrained_prior": pretrained_prior,
                        "det_model_fine_tuning": det_model_fine_tuning,
                        },
    ssm_hparams={       
                        "ssm": ssm,
                        "primary_type": primary_type,
                        "secondary_type": secondary_type,
                        "tertiary_type": tertiary_type,
                        },
    other_hparams={
                        "linearize": linearize,
                        "output_var": output_var,
                        "stochastic": stochastic,
                        "evaluate": evaluate,
                        "restore_checkpoint": restore_checkpoint,
                        "batch_stats_init_epochs": batch_stats_init_epochs,
                        "dataset": dataset,
                        "prediction_type": prediction_type,
                        "mimic_task": mimic_task,
                        "ood_points": ood_points,
                        "inference_data_model_name": inference_data_model_name,
                        "context_batch_size": context_batch_size,
                        "context_dataset_size": context_dataset_size,
                        "num_epochs": num_epochs,
                        "seed": seed,
                        "jobid": jobid,
                        "mc_samples_eval": mc_samples_eval,
                        "config_name": config_name,
                        "debug_print": debug_print,
                        "log_frequency": log_frequency,
                        "final_eval": final_eval,
                        "eval_patient_groups": eval_patient_groups,
                        "save_model_on": save_model_on,
                        "save_to_wandb": save_to_wandb,
                        "wandb_project": wandb_project,
                        "wandb_account": wandb_account,
                        "confidence_intervals": confidence_intervals,
                        "label_file_splits": label_file_splits,
                        "full_training": full_training,
                        "model_for_final_eval": model_for_final_eval,
                        "pretrained_prior_path": pretrained_prior_path,
                        "build_context_set": build_context_set,
                        "set_zero_gradients": set_zero_gradients,
                        "save_plot_data": save_plot_data
                        },
    exmp_inputs=jax.device_put(
        next(iter(train_loader))[0]),
    rng_key=rng_key,
    )

print(f"\nDone\n")