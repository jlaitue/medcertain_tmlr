import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from random import choice

class EHRdataset(Dataset):
    def __init__(self, discretizer, normalizer, listfile, dataset_dir, split="train", return_names=True, transform=False, transform_hypers = None, period_length=48.0):
        self.return_names = return_names
        self.discretizer = discretizer
        self.normalizer = normalizer
        self._period_length = period_length
        self.transform = transform
        self.transform_hypers = transform_hypers

        if split == "test":
            self.ehr_folder = "test"
        else:
            self.ehr_folder = "train"

        self.base_dir = dataset_dir
        self._dataset_dir = self.base_dir + self.ehr_folder
        listfile_path = listfile

        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES = self._listfile_header.strip().split(",")[3:]
        self._data = self._data[1:]

        self._data = [line.split(",") for line in self._data]
        self.data_map = {
            mas[0]: {
                "labels": list(map(float, mas[3:])),
                "stay_id": float(mas[2]),
                "time": float(mas[1]),
                }
                for mas in self._data
        }

        self.names = list(self.data_map.keys())
    
    def _read_timeseries(self, ts_filename, time_bound=None):
        
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(",")
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(",")
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_by_file_name(self, index, time_bound=None):
        t = self.data_map[index]["time"] if time_bound is None else time_bound
        y = self.data_map[index]["labels"]
        stay_id = self.data_map[index]["stay_id"]
        # ----> (X, header) = self._read_timeseries(index, time_bound=time_bound)

        # THIS CODE IS USED FOR SEED SPLITS. If you want to go to the original code
        # comment until the dashed line and uncomment lines with ARROWS.
        # However, this code can be used for loading the seed split or original files.
        try:
            (X, header) = self._read_timeseries(index, time_bound=time_bound)
        except:
            # We looked for our EHR csv files in the train/ folder in the server and failed,
            # we have to change to the test/ folder since the seed splits shuffle files from both folders.
            if "train" in self._dataset_dir:
                self._dataset_dir = self.base_dir + "test"
            else:
                self._dataset_dir = self.base_dir + "train"
            (X, header) = self._read_timeseries(index, time_bound=time_bound)
        #---------------------------------------------------------------------
            
        return {"X": X, "t": t, "y": y, "stay_id": stay_id, "header": header, "name": index}

    def get_decomp_los(self, index, time_bound=None):
        return self.__getitem__(index, time_bound)

    # -------------------------------
    # TRANSFORMS for context dataset
    # TODO: drop_start Only works with phenotyping, this should only mask the beginning with zeros not drop the cols
    def drop_start(self, data, max_percent=0.4):
        length = data.shape[0]
        start = int(np.random.randint(low=0, high=max(int(max_percent*length),1), size=1))
        return data[start:,:]
        # return data
    
    def gaussian_blur(self, data, mean=1, std=0, max_columns=76):
        data[:, :max_columns] = data[:, :max_columns]  + np.random.normal(mean, std, (data.shape[0], max_columns))
        return data

    def rotation(self, data):
        if choice([0,1]):
            return np.flip(data, axis=0)
        return data
    #-----------------------------------
    
    def __getitem__(self, index, time_bound=None):
        if isinstance(index, int):
            index = self.names[index]

        # THIS CODE IS USED FOR SEED SPLITS. If you want to go to the original code
        # comment until the dashed line and uncomment lines with ARROWS.
        try:
            ret = self.read_by_file_name(index, time_bound)
        except:
            # We looked for our EHR csv files in the train/ folder in the server and failed,
            # we have to change to the test/ folder since the seed splits shuffle files from both folders.
            if self.ehr_folder == "train":
                self._dataset_dir = self.base_dir + "test"
            else:
                self._dataset_dir = self.base_dir + "train"
            ret = self.read_by_file_name(index, time_bound)

        # ---> ret = self.read_by_file_name(index, time_bound)
        data = ret["X"]
        ts = ret["t"] if ret['t'] > 0.0 else self._period_length
        ys = ret["y"]
        names = ret["name"] # Original MedFuse. This is not used, I guess it should be self.names?
        data = self.discretizer.transform(data, end=ts)[0]
        #---------------------------------------------------------------------
        # SHOULD THIS GO BEFORE OR AFTER NORMALIZATION?
        if self.transform:
            # data = self.drop_start(data, max_percent=self.transform_hypers["dropstart_max_percent"]) # Not using drop start for the moment
            data = self.gaussian_blur(data, mean=self.transform_hypers["gaussian_mean"], std=self.transform_hypers["gaussian_std"], max_columns=self.transform_hypers["gaussian_max_cols"])
            data = self.rotation(data)
        #---------------------------------------------------------------------
        if (self.normalizer is not None): # Shouldnt this only normalize based on training? What is happening here?
            data = self.normalizer.transform(data)
        
        ys = np.array(ys, dtype=np.int32) if len(ys) > 1 else np.array(ys, dtype=np.int32)[0]
        return data, ys

    
    def __len__(self):
        return len(self.names)


def get_datasets(discretizer, normalizer, args, transform=False, transform_hypers=None):
    print("LOADING MIMIC-EHR DATASET")

    if args["label_file_splits"] != "original": # Any of: original | seed_test | same_test | medfuse_test
        type_test = args["label_file_splits"]
        print(f"LOADING MODIFIED TRAIN VAL SPLIT with SEED {args['seed']} and flag {type_test}")
        train_set_listfile_path = f"data/MedFuse/{args['task']}/train_listfile_seed_{args['seed']}_{type_test}.csv"
        val_set_listfile_path = f"data/MedFuse/{args['task']}/val_listfile_seed_{args['seed']}_{type_test}.csv"

        test_set_listfile_path = ""
        if "same_test" in type_test:
            test_set_listfile_path = f"data/MedFuse/{args['task']}/test_listfile_{type_test}.csv"
        elif "medfuse_test" in type_test:
            test_set_listfile_path = f"{args['ehr_data_dir']}/{args['task']}/test_listfile.csv"
        else:
            test_set_listfile_path = f"data/MedFuse/{args['task']}/test_listfile_seed_{args['seed']}_{type_test}.csv"
    else:
        # Else, we load the original csv files from the server that contains the MIMIC dataset
        train_set_listfile_path = f"{args['ehr_data_dir']}/{args['task']}/train_listfile.csv"
        val_set_listfile_path = f"{args['ehr_data_dir']}/{args['task']}/val_listfile.csv"
        test_set_listfile_path = f"{args['ehr_data_dir']}/{args['task']}/test_listfile.csv"
    
    print(f"Train labels file: {train_set_listfile_path}")
    print(f"Val labels file: {val_set_listfile_path}")
    print(f"Test labels file: {test_set_listfile_path}\n")

    train_ds = EHRdataset(discretizer, normalizer, train_set_listfile_path, f"{args['ehr_data_dir']}/{args['task']}/", split="train", transform=transform, transform_hypers=transform_hypers)
    val_ds = EHRdataset(discretizer, normalizer, val_set_listfile_path, f"{args['ehr_data_dir']}/{args['task']}/", split="val")
    test_ds = EHRdataset(discretizer, normalizer, test_set_listfile_path, f"{args['ehr_data_dir']}/{args['task']}/", split="test")
    
    return train_ds, val_ds, test_ds
        
def my_collate(batch):
    x = [item[0] for item in batch]   
    targets = np.array([item[1] for item in batch])
    task = "phenotyping"
    if targets.shape[-1] != 25: # MORTALITY task
        task = "in-hospital-mortality"
        # Only in task MORTALITY targets are sent as (n, ) 
        # a 1-dimensional array and we want to send them as (n,1)
        targets = np.expand_dims(targets, axis=-1)
    x, _ = pad_zeros(x, task, None)
    return (x, targets)

def pad_zeros(arr, task, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]

    if task == "in-hospital-mortality":
        max_len = max(seq_length)
        ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in arr]
    else: # CLINICAL CONDITION
        max_len = 3000
        ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0) for x in arr]
        ret = [x[:100, :] for x in ret]

    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length