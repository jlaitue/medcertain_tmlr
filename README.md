# Learned Data-driven Priors for Uncertainty-Aware Deterioration Risk Prediction with Multimodal Data

Table of contents
=================

  * [Background](#Background)
  * [Directory overview](#Directory-overview)
  * [Getting started](#Getting-started)
  <!-- * [License](#License) -->


Background
============
Model safety is a crucial requirement for integrating predictive models into clinical decision support systems. 
One approach for ensuring trustworthy use is the ability to measure uncertainty in individual predictions. 
However, current machine learning models frequently lack reliable uncertainty quantification, hindering real-world deployment. This is further observed in multimodal settings, where the goal is merely effective information fusion.
In this work, we propose MedCertAIn, a predictive uncertainty framework that leverages multimodal clinical data for in-hospital risk prediction to improve both model performance and reliability. We design data-driven priors over neural network parameters using a hybrid strategy that leverages inter-modal similarity in self-supervised latent representations and raw data corruptions.
We train and evaluate our proposed approach using clinical time-series and chest X-ray images from the publicly-available datasets MIMIC-IV and MIMIC-CXR. 
Our results show that MedCertAIn significantly enhances predictive performance and uncertainty quantification compared to state-of-the-art deterministic baselines and existing Bayesian methods. 
These findings highlight the promise of data-driven priors in advancing robust, uncertainty-aware AI tools for high-stakes clinical applications.

Directory overview
====================================

### MedFuse/
- Contains the main MedFuse model architectures.

### configs/
- Configuration files for different training methods, unimodal and multimodal as well as hyperparameters for the group aware priors framework.

### processing_scripts/
- Scripts for processing images, datasets, and evaluation metrics.

### shell_scripts/ and job_files/
- Scripts for submitting batch jobs to an HPC cluster.

### arguments.py
- Parses arguments sent through the terminal.

### base_architectures.py and base_datasets.py
- Base code to loading and processing datasets and model weights.

### functions.py
- Contains various utility functions used throughout the code for calculating performance metrics and custom loss functions used in model training.

### trainer.py
- Main script.
- **Functionality**:
  - **Data Loading**: Prepares data for training and evaluation.
  - **Model Initialization**: Sets up the LSTM and ResNet models.
  - **Training Loops**: Core training routines.
  - **Evaluation**: Performance assessment of trained models.
  - **Optimizers**: Configuration of optimization algorithms.
  - **Data Extraction During Inference**: Extracts data for analysis during model inference.

Getting started
====================================
TODO: Currently drafting specific instructions to provide an easy-to-follow setup, training and evaluation.


<!-- License
====================================
This project is licensed under the MIT License. See the LICENSE file for details. -->
