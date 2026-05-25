# Data-Driven Priors for Uncertainty-Aware Deterioration Risk Prediction with Multimodal Data

Minimal instructions for training and evaluating MedFuse, the uninformative-prior baseline, and MedCertAIn on MIMIC-CXR/EHR fusion data.

## 1. Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate medcertain
```

Run all commands from the repository root unless your cluster scripts do something different.

## 2. Set local paths

Before running anything, edit `utils/constants.py`:

```python
DATASET_PATH = "data"
CHECKPOINT_PATH = "checkpoints"

MIMIC_CXR_PATH = "data/physionet.org/files/mimic-cxr-jpg/2.0.0/"
MIMIC_EHR_PATH = "data/mimic-iv-extracted/"
MIMIC_PATIENT_METADATA_FILE = "data/physionet.org/files/mimic-iv-1.0/core/patients.csv"
```

Set these to the locations on your machine/cluster:

- `MIMIC_CXR_PATH`: MIMIC-CXR-JPG root directory.
- `MIMIC_EHR_PATH`: extracted MIMIC-IV/EHR directory used by MedFuse.
- `MIMIC_PATIENT_METADATA_FILE`: MIMIC-IV `patients.csv` file.
- `CHECKPOINT_PATH`: where trained models/checkpoints are saved and loaded from.
- `DATASET_PATH`: root for any auxiliary data used by the dataloaders.

If you save inference outputs with `--save_plot_data`, also make sure the inference output directory exists. In the current code this is written under:

```bash
data/MedFuse/inference_data/<mimic_task>/
```

For example:

```bash
mkdir -p data/MedFuse/inference_data/phenotyping
```

## 3. Code entry point

The main entry point is `main.py`. It handles the high-level pipeline:

1. parse args and configure the run,
2. load MIMIC fusion data,
3. initialize the MedCertAIn module,
4. train if `--evaluate` is not set,
5. load a checkpoint,
6. evaluate on the test set or save inference outputs.

Core implementation details are in `utils/` and `MedFuse/`.

## 4. Recommended pipeline

The expected workflow is:

1. Train deterministic MedFuse for 5 seeds.
2. Use those MedFuse checkpoints to initialize MedCertAIn.
3. Fine-tune MedCertAIn from the corresponding MedFuse checkpoint.
4. Evaluate the trained models on the test set.

The shell entry points are organized as:

- `shell_scripts/train/medfuse.sh`
- `shell_scripts/train/uninformative_prior.sh`
- `shell_scripts/train/medcertain.sh`
- `shell_scripts/eval/medfuse.sh`
- `shell_scripts/eval/uninformative_prior.sh`
- `shell_scripts/eval/medcertain.sh`

## 5. Training scripts

Use the scripts in `shell_scripts/train/` for training runs. Update the script files with your cluster/local command details as needed.

- `shell_scripts/train/medfuse.sh`: train deterministic MedFuse from scratch for 5 seeds.
- `shell_scripts/train/uninformative_prior.sh`: train the uninformative-prior Bayesian baseline.
- `shell_scripts/train/medcertain.sh`: fine-tune MedCertAIn initialized from the trained MedFuse checkpoints.

The intended MedCertAIn workflow is to first train MedFuse for seeds `0`-`4`, then use those trained MedFuse checkpoints to initialize and fine-tune the corresponding MedCertAIn models.

## 6. Evaluation scripts

Use the scripts in `shell_scripts/eval/` for test-set evaluation. For evaluating a specific saved model folder, update the relevant script so `--model_for_final_eval` points to that checkpoint folder on disk.

- `shell_scripts/eval/medfuse.sh`: evaluate deterministic MedFuse.
- `shell_scripts/eval/uninformative_prior.sh`: evaluate the uninformative-prior Bayesian baseline.
- `shell_scripts/eval/medcertain.sh`: evaluate MedCertAIn.

`--model_for_final_eval` accepts:

- `BEST`: load the best checkpoint from the current run directory.
- `LAST`: load the last checkpoint from the current run directory.
- `/path/to/checkpoint/folder`: load a specific checkpoint folder from disk. Use this for final evaluation of already-trained models.

To save predictions/targets for downstream plotting instead of computing the full metrics, add the relevant inference flags in the evaluation script, including:

```bash
--save_plot_data
--inference_data_model_name <name_for_saved_outputs>
```

## 7. Config files

Example configs live in `configs/`. The most relevant fusion configs are:

```bash
configs/nn-tdvi-pt-fusion-mimic-det.json      # deterministic MedFuse
configs/nn-tdvi-pt-fusion-mimic-mfvi.json     # uninformative-prior baseline
configs/nn-tdvi-pt-fusion-mimic-psvi.json     # MedCertAIn
```

Use the shell scripts/configs as the run interface. For final evaluation, the most important flag to set inside the evaluation script is `--model_for_final_eval`.

## Data availability

The MIMIC-CXR/MIMIC-IV datasets must be obtained separately through PhysioNet and are not included in this repository.

The MedCertAIn context-set artifacts used for the main experiments are also not included because of their large file size. These files can be shared upon request. After obtaining them, place them at the paths expected by `utils/constants.py` and the corresponding shell scripts.