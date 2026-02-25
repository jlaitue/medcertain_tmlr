import numpy as np
from torch.utils.data import Dataset
from random import choice
from torchvision import transforms
from PIL import Image

class FusionContextDataset(Dataset):
    def __init__(self, args, ehr_ds, cxr_ds, ehr_cxr_pairs, points="misclassified", corrupted=False, merged=False, ehr_corrupt_hypers=None, cxr_corrupt_transforms=None):
        
        self.ehr_inputs = ehr_ds[0]
        self.ehr_targets = ehr_ds[1]
        
        self.cxr_inputs = cxr_ds[0]
        self.cxr_targets = cxr_ds[1]

        self.ehr_cxr_pairs = ehr_cxr_pairs 

        self.args = args
        self.points = points

        self.corrupted = corrupted
        self.merged = merged
        self.ehr_corrupt_hypers = ehr_corrupt_hypers
        self.cxr_corrupt_transforms = cxr_corrupt_transforms
    #-----------------------------------
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

    def __getitem__(self, index):
        ehr_data = self.ehr_inputs[index]
        labels_ehr = self.ehr_targets[index]
        cxr_data = self.cxr_inputs[index]
        labels_cxr = self.cxr_targets[index]
        ehr_cxr_pairs = self.ehr_cxr_pairs[index]
        if self.corrupted:
            ehr_data = self.gaussian_blur(ehr_data, mean=self.ehr_corrupt_hypers["gaussian_mean"], std=self.ehr_corrupt_hypers["gaussian_std"], max_columns=self.ehr_corrupt_hypers["gaussian_max_cols"])
            ehr_data = self.rotation(ehr_data)
            # Had to do this because the vision transforms require PIL images
            cxr_data = Image.fromarray(cxr_data.astype(np.uint8)).convert("RGB")
            cxr_data = self.cxr_corrupt_transforms(cxr_data)
        # TODO: This needs to be addressed in the original ehr data shape for mortality
        if self.merged and self.args["task"] == "in-hospital-mortality":
            # In mortality: If we merged the latent context set with the corrupted train 
            # context set the ehr labels come in different shapes: (n,1) and (n, ) respectively. 
            # This ensures we have no issue in the concat operation when the Dataloader 
            # uses the colllate fn.
            labels_ehr = labels_ehr.mean(axis=0)
        return ehr_data, cxr_data, labels_ehr, labels_cxr, ehr_cxr_pairs
    
    def __len__(self):
        return len(self.ehr_inputs)

def image_to_numpy(img):
    DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    DATA_STD = np.array([0.247, 0.243, 0.261])
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    return img  

def cxr_transforms(args):
    cxr_corrupt_hypers = args["cxr_corrupt_hypers"]
    gaussian_blur_kernel = (cxr_corrupt_hypers["gaussianblur_kernel"], cxr_corrupt_hypers["gaussianblur_kernel"])
    
    corrupt_transform_list = [
        transforms.RandomCrop(size=cxr_corrupt_hypers["randcrop_size"], padding=4),
        transforms.RandomHorizontalFlip(cxr_corrupt_hypers["randhorizontalflip_prob"]), 
        transforms.RandomVerticalFlip(cxr_corrupt_hypers["randverticalflip_prob"]),
        transforms.GaussianBlur(kernel_size=gaussian_blur_kernel),
        transforms.RandomSolarize(threshold=cxr_corrupt_hypers["randsolarize_threshold"], p=cxr_corrupt_hypers["randsolarize_prob"]),
        transforms.RandomInvert(cxr_corrupt_hypers["randinvert_prob"]),
        transforms.ColorJitter(brightness=cxr_corrupt_hypers["colorjitter_brightness"], contrast=cxr_corrupt_hypers["colorjitter_contrast"]),
        transforms.Resize(224), # We need 224x224 for MIMIC-CXR
        ]
    
    corrupt_transform_list.append(image_to_numpy)
    corrupt_transform = transforms.Compose(corrupt_transform_list)
    return corrupt_transform

def load_context_dataset(args):

    print(f"LOADING MIMIC-FUSION CONTEXT DATASET. Points: {args['points']} | Corrupted: {args['corrupted']}")

    npz_file = f"data/MedFuse/ContextPoints/{args['data_points_file']}"
    print(f" Loading context file: {npz_file}")
    context_dataset_dict = np.load(npz_file)
    ehr_ds = (context_dataset_dict["ehr_inputs"], context_dataset_dict["ehr_targets"])
    cxr_ds = (context_dataset_dict["cxr_inputs"], context_dataset_dict["cxr_targets"])
    ehr_cxr_pairs = context_dataset_dict["ehr_cxr_pairs"]
    points = args["points"]
    cxr_corrupt_transforms = cxr_transforms(args)
    context_ds = FusionContextDataset(args, ehr_ds, cxr_ds, ehr_cxr_pairs, points, corrupted=args["corrupted"], merged=args["merged"], ehr_corrupt_hypers=args["ehr_corrupt_hypers"], cxr_corrupt_transforms=cxr_corrupt_transforms)
    return context_ds

def my_collate(batch):
    x = np.array([item[0] for item in batch])
    img = np.array([item[1] for item in batch])
    targets_ehr = np.array([item[2] for item in batch])
    targets_cxr = np.array([item[3] for item in batch])
    pairs = np.array([item[4] for item in batch])
    return [(x, targets_ehr, img, targets_cxr, pairs)]