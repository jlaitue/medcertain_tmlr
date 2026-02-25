from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from typing import (Any, Callable, Iterable, Optional, Tuple, Union, Dict)
from functions import numpy_collate
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, MNIST, KMNIST, ImageNet
from sklearn import datasets as sklearn_datasets
from MedFuse.datasets import cxr_dataset, ehr_dataset, fusion_dataset
from MedFuse.ehr_utils.preprocessing import Discretizer, Normalizer
import os
import glob

DATASET_PATH = "data"

class CustomDataset(Dataset):
    def __init__(self, original_dataset, desired_size):
        self.original_dataset = original_dataset
        self.desired_size = desired_size

    def __len__(self):
        return self.desired_size

    def __getitem__(self, idx):
        idx = idx % len(self.original_dataset)  # wrap around the original dataset
        return self.original_dataset[idx]

# ------------------------------------------------------------------------
# MIMIC EHR FUNCTIONS
def read_timeseries(args):
        path = f"{args['ehr_data_dir']}/{args['task']}/train/14991576_episode3_timeseries.csv"
        ret = []
        with open(path, "r") as tsfile:
            header = tsfile.readline().strip().split(",")
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(",")
                ret.append(np.array(mas))
        return np.stack(ret)

def get_discretizer(args):
    discretizer = Discretizer(timestep=float(args["timestep"]),
                              store_masks=True,
                              impute_strategy="previous",
                              start_time="zero")
    return discretizer

def get_normalizer(discretizer, args):
    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(",")
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args["normalizer_state"]

    if normalizer_state is None:
        normalizer_state = f"MedFuse/normalizers/ph_ts{args['timestep']}.input_str:previous.start_time:zero.normalizer"
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    return normalizer

def load_MIMIC_EHR(mimic_ehr_args):
    discretizer = get_discretizer(mimic_ehr_args)
    normalizer = get_normalizer(discretizer, mimic_ehr_args)

    train_dataset, validation_dataset, test_dataset = ehr_dataset.get_datasets(discretizer, normalizer, mimic_ehr_args)

    training_dataset_size = len(train_dataset)
    validation_dataset_size = len(validation_dataset)
    test_dataset_size = len(test_dataset)

    return train_dataset, validation_dataset, test_dataset, training_dataset_size, validation_dataset_size, test_dataset_size
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# MIMIC CXR FUNCTIONS
def load_MIMIC_CXR(mimic_cxr_args):
    print("LOADING MIMIC-CXR DATASET")
    train_transforms, test_transforms = cxr_dataset.get_transforms(mimic_cxr_args)
    paths = glob.glob(f"{mimic_cxr_args['cxr_data_dir']}resized/**/*.jpg", recursive = True)

    train_dataset = cxr_dataset.MIMICCXR(paths, mimic_cxr_args, split="train", transform=train_transforms)
    if mimic_cxr_args["full_training"]:
        validation_dataset = cxr_dataset.MIMICCXR(paths, mimic_cxr_args, split="validate", transform=train_transforms) # We use the train transforms
    else:
        validation_dataset = cxr_dataset.MIMICCXR(paths, mimic_cxr_args, split="validate", transform=test_transforms)  # We use the test transforms
    test_dataset = cxr_dataset.MIMICCXR(paths, mimic_cxr_args, split="test", transform=test_transforms)

    training_dataset_size = len(train_dataset)
    validation_dataset_size = len(validation_dataset)
    test_dataset_size = len(test_dataset)

    return train_dataset, validation_dataset, test_dataset, training_dataset_size, validation_dataset_size, test_dataset_size, paths
