import os
import numpy as np
from PIL import Image
import pandas as pd 
import glob

## Pytorch
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# These are used with labels file "mimic-cxr-2.0.0-chexpert.csv"
R_CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices']

# These are used with labels file "mimic-cxr-pheno.csv"
CLASSES = ['Acute and unspecified renal failure', 'Acute cerebrovascular disease',
            'Acute myocardial infarction', 'Cardiac dysrhythmias',
            'Chronic kidney disease',
            'Chronic obstructive pulmonary disease and bronchiectasis',
            'Complications of surgical procedures or medical care',
            'Conduction disorders', 'Congestive heart failure; nonhypertensive',
            'Coronary atherosclerosis and other heart disease',
            'Diabetes mellitus with complications',
            'Diabetes mellitus without complication',
            'Disorders of lipid metabolism', 'Essential hypertension',
            'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
            'Hypertension with complications and secondary hypertension',
            'Other liver diseases', 'Other lower respiratory disease',
            'Other upper respiratory disease',
            'Pleurisy; pneumothorax; pulmonary collapse',
            'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
            'Respiratory failure; insufficiency; arrest (adult)',
            'Septicemia (except in labor)', 'Shock'
            ]
            
class MIMICCXR(Dataset):
    def __init__(self, paths, args, transform=None, split="train"):
        self.data_dir = args["cxr_data_dir"]
        self.args = args
        self.CLASSES  = CLASSES
        
        if args["labels_file"] == "mimic-cxr-2.0.0-chexpert.csv":
            self.CLASSES  = R_CLASSES

        self.transform = transform
        self.filenames_to_path = {os.path.basename(path).split(".")[0]: path for path in paths}
    
        labels = pd.read_csv(self.data_dir + args["labels_file"])
        labels[self.CLASSES] = labels[self.CLASSES].fillna(0)
        labels = labels.replace(-1.0, 0.0)

        splits = pd.read_csv(self.data_dir + args["split_file"])

        metadata = pd.read_csv(self.data_dir + args["metadata_file"])
        metadata_with_labels = metadata.merge(labels[self.CLASSES+["study_id"]], how="inner", on="study_id")
        self.filesnames_to_labels = dict(zip(metadata_with_labels["dicom_id"].values, metadata_with_labels[self.CLASSES].values))

        # --------------------------------------------------------------------
        # CODE for seed dependent split files
        # We load all rows regardless of their original split belonging (train/validate/test) to be able to use
        # our seed dependent split files
        # This works for both original and seed dependent splits because we are basically
        # loading all CXR image names when we create the train val and test sets.
        self.filenames_loaded = splits["dicom_id"].values
        # --------------------------------------------------------------------

        # Uncomment these lines to go back to the original setup
        # self.filenames_loaded = splits.loc[splits.split == split]["dicom_id"].values
        # --------------------------------------------------------------------
        self.filenames_loaded = [filename for filename in self.filenames_loaded if filename in self.filesnames_to_labels]
        
    def __getitem__(self, index):
        if isinstance(index, str):
            img = Image.open(self.filenames_to_path[index]).convert("RGB")
            labels = np.array(self.filesnames_to_labels[index],  dtype = float)
        else:
            filename = self.filenames_loaded[index]
            img = Image.open(self.filenames_to_path[filename]).convert("RGB")
            labels = np.array(self.filesnames_to_labels[filename],  dtype = float)

        if self.transform is not None:
            img = self.transform(img)
        
        return (img, labels)
    
    def __len__(self):
        return len(self.filenames_loaded)

def image_to_numpy(img):
        DATA_MEANS = np.array([0.485, 0.456, 0.406])
        DATA_STD = np.array([0.229, 0.224, 0.225])
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

def get_transforms(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transforms = []
    train_transforms.append(transforms.Resize(256))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(224))
    train_transforms.append(image_to_numpy)       

    test_transforms = []
    test_transforms.append(transforms.Resize(args["resize"]))
    test_transforms.append(transforms.CenterCrop(args["crop"]))
    test_transforms.append(image_to_numpy)

    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    return train_transforms, test_transforms