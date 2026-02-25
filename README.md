<!-- # 🩻 Class-Dependent Miscalibration Severely Degrades Selective Prediction in Multimodal Clinical Prediction Models -->


Table of contents
=================

  * [Background](#Background)
  * [Directory overview](#Directory-overview)
  * [Getting started](#Getting-started)
  * [License](#License)


Background
============
As artificial intelligence systems move toward clinical deployment, ensuring reliable prediction behavior is fundamental for safety-critical decision-making tasks.
One proposed safeguard is selective prediction, where models can defer uncertain predictions to human experts for review.
In this work, we empirically evaluate the reliability of uncertainty-based selective prediction in multilabel clinical condition classification using multimodal ICU data.
Across a range of state-of-the-art unimodal and multimodal models, we find that selective prediction can substantially degrade performance despite strong standard evaluation metrics.
This failure is driven by severe class-dependent miscalibration, whereby models assign high uncertainty to correct predictions and low uncertainty to incorrect ones, particularly for underrepresented clinical conditions.
Our results show that commonly used aggregate metrics can obscure these effects, limiting their ability to assess selective prediction behavior in this setting.
Taken together, our findings characterize a task-specific failure mode of selective prediction in multimodal clinical condition classification and highlight the need for calibration-aware evaluation to provide strong guarantees of safety and robustness in clinical AI.

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

To get started, follow the instructions below.

### Prerequisites

- Python 3.8+
- Conda environment (recommended)

### Installation

1. Clone the repository:
   ```sh
   git clone https://anonymous.4open.science/r/medcalibration-B187/README.md

2. Create conda environment:

    conda env create -f environment.yml
    conda activate uq-wq

3. Refer to **shell_scripts/** and **job_files/** folders for specific cases of model training


License
====================================
This project is licensed under the MIT License. See the LICENSE file for details.
