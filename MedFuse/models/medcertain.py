import warnings

warnings.filterwarnings(
    "ignore",
    message="distutils Version classes are deprecated.*",
    category=DeprecationWarning,
)

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

## JAX
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit

## Flax
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze, unfreeze

## JAX addons
import optax
import neural_tangents as nt
from flaxmodels import utils

## Tensorflow
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
tfd = tfp.distributions

## PyTorch
import torch
import torch.utils.data as data
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve, confusion_matrix, accuracy_score
from pathlib import Path


## MedFuse code
from MedFuse.models import lstm_jax, fusion_jax
from MedFuse.datasets import cxr_dataset, ehr_dataset, fusion_dataset
from MedFuse.datasets import context_dataset as context_dataset_class
from MedFuse.ehr_utils.preprocessing import Discretizer, Normalizer
import MedFuse.confidence_intervals.ci_processing as medfuse_ci
import MedFuse.confidence_intervals.ci_processing_average as medfuse_ci_2

#-----------------------------------------------------------------------------

from utils.functions import (
    calibration, categorical_nll_with_softmax, gaussian_nll,
    mimic_ce_loss_sigmoid, upscaled_mimic_ce_loss_sigmoid,
    selective_prediction_mimic,
    merge_params, split_params, numpy_collate,
    dual_vector
    )

from utils.base_architectures import (
    ResNetBlock, ResNet, ResNet18, ResNet34,
    ResNet50, ResNet101, ResNet152    
)

from utils.constants import *

# JAX TRAINER
class TrainState(train_state.TrainState):
    batch_stats: Any

class MedcertainModule:
    def __init__(self,
                model_name:         str,
                model_class:        nn.Module,
                optimizer_name:     str,
                model_hparams:      dict,
                optimizer_hparams:  dict,
                objective_hparams:  dict,
                ssm_hparams:        dict,
                other_hparams:      dict,
                general_args:       dict,
                exmp_inputs:        Any,
                train_loader:       Any,
                val_loader:         Any,
                test_loader:        Any,
                context_loader:     Any,
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
        self.upscale_mimic_loss = objective_hparams["upscale_mimic_loss"]
        self.gap_regularizer = objective_hparams["gap_regularizer"]
        self.rho_sam = objective_hparams["rho_sam"]
        self.reg_type = self.objective_hparams["reg_type"]
        self.init_logvar = objective_hparams["init_logvar"]
        self.init_final_layer_weights_logvar = objective_hparams["init_final_layer_weights_logvar"]
        self.init_final_layer_bias_logvar = objective_hparams["init_final_layer_bias_logvar"]
        self.prior_feature_logvar = objective_hparams["prior_feature_logvar"]
        self.pretrained_prior = objective_hparams["pretrained_prior"]
        self.det_model_fine_tuning = objective_hparams["det_model_fine_tuning"]
        self.save_plot_data = other_hparams["save_plot_data"]

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
        
        self.training_dataset_size = len(train_loader.dataset)
        self.val_dataset_size = len(val_loader.dataset)
        self.test_dataset_size = len(test_loader.dataset)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.context_loader = context_loader

        self.n_batches_train = self.training_dataset_size / self.batch_size
        self.num_classes = self.model_hparams["num_classes"]
        self.inference_data_model_name = self.model_hparams["inference_data_model_name"]
        self.mc_dropout = self.objective_hparams["mc_dropout"]
        self.inference_data_model_name = f"{self.inference_data_model_name}_{self.mimic_task}_seed_{self.seed}"

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
            elif self.mimic_task in ["in-hospital-mortality", "1-year-in-hospital-mortality", "6-month-in-hospital-mortality", "3-month-in-hospital-mortality", "los_7_days"]:
                self.model = fusion_jax.Fusion(num_classes=1, vision_num_classes=1, labels_set="mortality", mc_dropout=self.mc_dropout)
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

        if general_args["method"] == "psvi":
            self.run_name = f'{exprt_time}_{general_args["model_name"]}_BNN_{general_args["num_epochs"]}_{general_args["lr"]}_{general_args["reg_type"]}_ctxtbsize:{general_args["context_batch_size"]}_prlkcovs_{general_args["prior_likelihood_cov_scale"]}_exprt_{general_args["jobid"]}_seed_{general_args["seed"]}_{self.mimic_task}_{self.label_file_splits}'
        if general_args["method"] == "psmap":
            self.run_name = f'{exprt_time}_{general_args["model_name"]}_DET_{general_args["num_epochs"]}_{general_args["lr"]}_{general_args["batch_size"]}_{general_args["alpha"]}_{general_args["exponent"]}_exprt_{general_args["jobid"]}_seed_{general_args["seed"]}_{self.mimic_task}_{self.label_file_splits}'
        if self.full_training:
            self.run_name += "_full_training"
            self.save_model_on = "None"
        if self.mc_dropout:
            self.run_name += "_mc_dropout"
        if self.gap_regularizer:
            self.run_name += f"_gap_{self.reg_scale_gap}_rhos_{self.rho_sam}"
        # --------------------------------------------------------------------


        # --------------------------------------------------------------------
        # TRAINING LOGGER
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.run_name)
        self.logger = make_logger()
        self.wandb_logger = []
        # --------------------------------------------------------------------

        self.create_functions()
        if (self.dataset == "mimic-cxr" or self.dataset == "mimic-ehr") and (self.mimic_task in ["in-hospital-mortality", "1-year-in-hospital-mortality", "6-month-in-hospital-mortality", "3-month-in-hospital-mortality", "los_7_days", "phenotyping"]):
            if self.dataset == "mimic-cxr":
                self.init_model(exmp_inputs[2])
            elif self.dataset == "mimic-ehr":
                self.init_model(exmp_inputs[0])
        else:
            self.init_model(exmp_inputs)
        # print(self.model.tabulate(random.PRNGKey(0), x=exmp_inputs[0]))

        if not self.stochastic and not self.mc_dropout:
            assert self.mc_samples_llk == 1
            assert self.mc_samples_eval == 1
        assert self.mc_samples_reg == 1 # if not ("fsmap" in method or "fsvi" in method) else True  # currently not implemented
        # assert self.objective_hparams["reg_points"] == "train" if self.objective_hparams["method"] == "psmap" else True

    def create_functions(self):
        # USED IN MEDFUSE
        def calculate_empirical_gaussian_prior_kl(params_samples, params_variational_mean, params_variational_logvar, preds_f, inputs, batch_stats, prior_likelihood_scale, prior_likelihood_f_scale, prior_likelihood_cov_scale, prior_likelihood_cov_diag, rng_key):
            # Fusion model requires specific rng_key with dropout key
            if self.prediction_type == "mimic":
                root_key = jax.random.PRNGKey(self.seed)
                params_rng, dropout_key  = jax.random.split(root_key, num=2)
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
                eps = jax.tree_map(lambda x: jax.random.normal(rng_key, x.shape), params_logvar)
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
            elif (self.dataset == "mimic-cxr" or self.dataset == "mimic-ehr") and (self.mimic_task in ["in-hospital-mortality", "1-year-in-hospital-mortality", "6-month-in-hospital-mortality", "3-month-in-hospital-mortality", "los_7_days", "phenotyping"]):
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
            elif self.reg_type == "empirical_gaussian_prior_kl":
                reg = calculate_empirical_gaussian_prior_kl(params_samples, params, params_logvar, preds_f_reg, inputs_reg, batch_stats, self.prior_likelihood_scale, self.prior_likelihood_f_scale, self.prior_likelihood_cov_scale, self.prior_likelihood_cov_diag, rng_key)
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

            if self.prediction_type == "mimic":
                logits = preds_f_llk
                num_labels = targets.shape[-1]
                preds = jnp.mean(jax.nn.sigmoid(logits), axis=0)

                if self.upscale_mimic_loss and num_labels == 25:
                    nll = jnp.sum(upscaled_mimic_ce_loss_sigmoid(preds, targets)) / num_labels
                else:
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
                raise ValueError(f"Prediction type is not supported {self.prediction_type}")

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
                        loader = enumerate(self.val_loader)
                    elif split == "test":
                        loader = enumerate(self.test_loader)
                    elif split == "train":
                        # --------------------------------------------------------------------
                        # Load train set to build the CONTEXT DATASET**
                        loader = enumerate(self.train_loader)
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
                    loader = enumerate(zip(self.test_loader, self.context_loader))
                    
                # for i, (batch, batch_2) in enumerate(zip(test_loader, context_loader)):
                for i, batch in loader:
                    if self.prediction_type == "mimic":
                        if (self.dataset == "mimic-cxr" or self.dataset == "mimic-ehr") and (self.mimic_task in ["in-hospital-mortality", "1-year-in-hospital-mortality", "6-month-in-hospital-mortality", "3-month-in-hospital-mortality", "los_7_days", "phenotyping"]):
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
                    logits_test = jnp.concatenate(_logits_test, axis=1)[:, :self.val_dataset_size, :]
                elif split == "train":
                    # Code used to create our CONTEXT DATASET**
                    logits_test = jnp.concatenate(_logits_test, axis=1)[:, :self.training_dataset_size, :]
                else:
                    logits_test = jnp.concatenate(_logits_test, axis=1)[:, :self.test_dataset_size, :]

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
                sel_auroc_ep1, _ = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUROC", model_name=f"{self.inference_data_model_name}", save_plot_data=save_plot_data)

                # Selective AUPRC
                sel_auprc_ep1, _ = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUPRC", model_name=f"{self.inference_data_model_name}", save_plot_data=save_plot_data)

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
                sel_auroc_ep1, _ = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUROC", model_name=f"{self.inference_data_model_name}", save_plot_data=save_plot_data)

                # Selective AUPRC
                sel_auprc_ep1, _ = selective_prediction_eval_protocol_1(targets_test, preds_test, eval_metric="AUPRC", model_name=f"{self.inference_data_model_name}", save_plot_data=save_plot_data)
                
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
        def calculate_metrics(params, params_logvar, rng_key, batch_stats, n_batches_eval, final_eval, build_context_set, save_plot_data):
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
                
                if save_plot_data:
                    np.savez(f"data/MedFuse/inference_data/{self.mimic_task}/{self.inference_data_model_name}.npz", 
                                model_name=self.model_name, 
                                # Logits
                                logits=logits_test.mean(axis=0),
                                # Preds
                                preds=preds_test,
                                # Targets
                                targets=targets_test,
                                )
                    print(f"Saved logits, preds and targets of loaded model on {split.upper()} set. SHAPE: {preds_test.shape}. Done.\n")

                if self.prediction_type == "mimic" and not build_context_set and not save_plot_data:
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

                        # Selective AUPRC
                        selective_auprc, _ = selective_prediction_mimic(targets_test, preds_test, eval_metric="AUPRC")
                        
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

        def eval_step(state, rng_key, n_batches_eval, final_eval, build_context_set, save_plot_data):
            calculate_metrics(state.params["params"], state.params["params_logvar"], rng_key, state.batch_stats, n_batches_eval, final_eval, build_context_set, save_plot_data)
    
        self.train_step = train_step
        self.evaluation_predictions = evaluation_predictions
        self.eval_step = eval_step
        self.pred_fn_jit = pred_fn_jit

    def init_model(self, exmp_inputs):
        print("Starting model initialization...")
        root_key = jax.random.PRNGKey(self.seed)
        params_rng, dropout_key  = jax.random.split(root_key, num=2)
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
        
        alpha = self.optimizer_hparams["alpha"]
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
                
                if self.save_to_wandb and self.prediction_type == "mimic":
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

            if self.save_to_wandb:
                wandb.log({'train_epoch_auroc': auroc, 'train_epoch_auprc': auprc})

            print(f"\n\nTRAIN EPOCH: {epoch} FINISHED.")
            print(f"\n AUROC: {auroc}  | AUPRC {auprc}")
            print("-------------------------------------------------------")

    def eval_model(self, rng_key, n_batches_eval, final_eval=False, build_context_set=False, save_plot_data=False):
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
        self.eval_step(self.state, rng_key, n_batches_eval, final_eval, build_context_set, save_plot_data)

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

    def get_best_model_path(self):
        if self.model_for_final_eval == "BEST":
            best_model_path = f"{self.log_dir}_True/{self.save_model_on}_checkpoint_best_0"
        elif self.model_for_final_eval == "LAST":
            best_model_path = f"{self.log_dir}_{self.num_epochs}/checkpoint_{self.num_epochs}"
        else: # Uses the path given in the args by the user
            best_model_path = self.model_for_final_eval

        return best_model_path

    def load_best_model_ckpt(self, best_model_path, rng_key):
        state_dict = freeze(checkpoints.restore_checkpoint(ckpt_dir=best_model_path, target=None))

        if self.stochastic:
            params_logvar = state_dict['params_logvar']
        else:
            params_logvar = None

        if self.linearize and self.pred_fn is None:
            NotImplementedError
        else:
            self.pred_fn = self.model.apply

        params = freeze({"params": state_dict['params'], "params_logvar": params_logvar, 'dropout': rng_key})

        self.state = TrainState.create(apply_fn=self.model.apply,
                                    params=params,
                                    batch_stats=state_dict['batch_stats'],
                                    tx=self.state.tx if self.state else optax.sgd(0.1)   # Default optimizer
                                    )
        print(f"\nFinished loading best model checkpoint:\n {best_model_path}")
        print("------------------------------------------------------------\n\n")
    
    def final_log_wandb(self):
        self.wandb_logger.append({})
    
        print("\nALL METRICS")
        pprint(self.wandb_logger[-1])
        
        wandb.log(self.wandb_logger[-1])
        time.sleep(10)