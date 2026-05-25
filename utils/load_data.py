import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from MedFuse.datasets import cxr_dataset, ehr_dataset, fusion_dataset
from MedFuse.datasets import context_dataset as context_dataset_class
from MedFuse.ehr_utils.preprocessing import Discretizer, Normalizer

from utils.constants import (
    DATASET_PATH, CHECKPOINT_PATH, MIMIC_CXR_PATH, 
    MIMIC_EHR_PATH, MIMIC_PATIENT_METADATA_FILE
    )

from utils.base_datasets import (
    CustomDataset,
    # MIMIC EHR
    get_discretizer, get_normalizer, load_MIMIC_EHR,
    # MIMIC CXR
    load_MIMIC_CXR
)

def get_mimic_radiology_datasets(args):

    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("Creating MIMIC-CXR dataset for RADIOLOGY task...")

    # We need to set this, since we use the same dataloaders as the other datasets
    batch_size_test = args["batch_size"]
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
        "full_training": args["full_training"]
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

    if args["full_training"]:
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

    if args["context_points"] == "train":
        context_dataset = cxr_dataset.MIMICCXR(paths, mimic_cxr_args, split="train", transform=context_transform)
    elif args["context_points"] == "svhn":
        context_dataset = SVHN(root=f"./{DATASET_PATH}/SVHN", split="train", transform=context_transform, download=False)
    elif args["context_points"] == "imagenet":
        context_dataset = ImageNet(root=f"./{DATASET_PATH}/ImageNet", train=True, transform=context_transform, download=False)
    else:
        ValueError("Unknown context dataset")

    full_context_dataset_size = context_dataset_size = len(context_dataset)
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, full_context_dataset_size - context_dataset_size], generator=torch.Generator().manual_seed(args["seed"]))
    context_set = CustomDataset(context_set, training_dataset_size)
    context_dataset_size = len(context_set)
    # ------------------------------------------------------------------------

    print(f" Datasets lengths\n Training: {training_dataset_size} | Validation: {val_dataset_size} | Testing: {test_dataset_size} | Context: {context_dataset_size}")
    print("CXR datasets done.")
    print()

    return train_dataset, validation_dataset, test_dataset, context_dataset

def get_mimic_fusion_datasets(args):
    
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("Creating FUSION-CXR-EHR datasets...")

    mimic_task = args["mimic_task"]
    full_training = args["full_training"]
    context_points = args["context_points"]
    cxr_context_hypers = args["cxr_context_hypers"]
    ehr_context_hypers = args["ehr_context_hypers"]
    
    context_points_data_file = args["context_points_data_file"]
    corrupted = args["corrupted"]
    label_file_splits = args["label_file_splits"]
    seed = args["seed"]

    context_batch_size = args["context_batch_size"]
    batch_size = args["batch_size"]
    

    if mimic_task == "phenotyping":
        num_classes = 25
        labels_set = "phenotyping"
        # labels_file = "mimic-cxr-pheno.csv"
        labels_file = "mimic-cxr-2.0.0-chexpert.csv"
    elif mimic_task in ["in-hospital-mortality", "1-year-in-hospital-mortality", "6-month-in-hospital-mortality", "3-month-in-hospital-mortality", "los_7_days"]:
        num_classes = 1
        # labels_set = "mortality" # This is not used anywhere either in this code or original Medfuse
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
        # "labels_set": labels_set,
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

    return train_dataset, validation_dataset, test_dataset, context_dataset

def get_mimic_data(args):

    if args["dataset"] == "mimic-cxr" and args["mimic_task"] == "radiology":
        train_dataset, validation_dataset, test_dataset, context_dataset = get_mimic_radiology_datasets(args)
    
    elif args["dataset"] == "mimic-fusion" or ((args["dataset"] == "mimic-cxr" or args["dataset"] == "mimic-ehr") and (args["mimic_task"] in ["in-hospital-mortality", "1-year-in-hospital-mortality", "6-month-in-hospital-mortality", "3-month-in-hospital-mortality", "los_7_days", "phenotyping"])):
        train_dataset, validation_dataset, test_dataset, context_dataset = get_mimic_fusion_datasets(args)

    else:
        raise ValueError("Dataset selected not found.")
    #-----------------------------------------------------------------------------
    # DATALOADERS
    # We use a different collate function for the dataloaders on the MIMIC-EHR and MIMIC-FUSION datasets.

    if args["context_points"] in ["train", "merged"]:
        context_collate_fn = fusion_dataset.my_collate
    elif args["context_points"] in ["misclassified", "latent"]:
        context_collate_fn = context_dataset_class.my_collate
    else:
        # Any other scenario we may not consider
        context_collate_fn = fusion_dataset.my_collate

    train_loader = data.DataLoader(train_dataset,
                                    batch_size=args["batch_size"],
                                    shuffle=True,
                                    collate_fn=fusion_dataset.my_collate,
                                    num_workers=args["num_workers"],
                                    persistent_workers=True)
    
    val_loader = data.DataLoader(validation_dataset, 
                                    batch_size=args["batch_size"],
                                    shuffle=False,
                                    collate_fn=fusion_dataset.my_collate,
                                    num_workers=args["num_workers"],
                                    persistent_workers=True)
    
    test_loader  = data.DataLoader(test_dataset,
                                    batch_size=args["batch_size"],
                                    shuffle=False,
                                    drop_last=False,
                                    collate_fn=fusion_dataset.my_collate,
                                    num_workers=args["num_workers"],
                                    persistent_workers=True)
    
    context_loader  = data.DataLoader(context_dataset,
                                    batch_size=args["context_batch_size"],
                                    shuffle=True,
                                    drop_last=False,
                                    collate_fn=context_collate_fn,
                                    num_workers=args["num_workers"],
                                    persistent_workers=True)

    print("FINAL DATA STATISTICS:\n")
    print(f" DATASETS\n Training: {len(train_dataset)} | Validation: {len(validation_dataset)} | Testing: {len(test_dataset)} | Context: {len(context_dataset)}")
    print()
    print(f" DATA LOADERS\n Training: {len(train_loader)} | Validation: {len(val_loader)} | Testing: {len(test_loader)} | Context: {len(context_loader)}")
    if len(train_loader) > len(context_loader):
        raise ValueError("The size of your train loader is larger than the context loader.")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")

    if args["full_training"]:
        print("\nBEGINNING FULL TRAINING (TRAIN+VAL)...\n") # Training with train+val sets for final model after finetuning
    else:
        print("\nBEGINNING EXPERIMENT...\n")

    return train_loader, val_loader, test_loader, context_loader
    #-----------------------------------------------------------------------------