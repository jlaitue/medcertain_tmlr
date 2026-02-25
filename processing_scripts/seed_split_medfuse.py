import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--task", default="in-hospital-mortality", help="MIMIC task. One of 'in-hospital-mortality' or 'phenotyping'")
parser.add_argument("-s", "--n_seeds", default=5, type=int, help="Number of seeds to run")
args = vars(parser.parse_args())
 
# Set up parameters
task = args["task"]
n_seeds = args["n_seeds"]
seeds = list(range(0, n_seeds))
mimic_ehr_path = "laboratory/data/mimic-iv-extracted/" # MIMIC EHR dataset location

print(f"Current settings for creation of splits\n{args}\n")
input("Press any key to continue...")

# Load original files from host of MIMIC dataset
train_labels = pd.read_csv(f"{mimic_ehr_path}/{task}/train_listfile.csv")
val_labels = pd.read_csv(f"{mimic_ehr_path}/{task}/val_listfile.csv")
test_labels = pd.read_csv(f"{mimic_ehr_path}/{task}/test_listfile.csv")

# We add a new column patient_id which is based on the first portion of the column "filename" of the dfs
train_labels["patient_id"] = train_labels.apply(lambda x: x.stay.split("_")[0], axis=1)
val_labels["patient_id"] = val_labels.apply(lambda x: x.stay.split("_")[0], axis=1)
test_labels["patient_id"] = test_labels.apply(lambda x: x.stay.split("_")[0], axis=1)

train_labels["original_split"], val_labels["original_split"] = "train", "val"
TRAIN_VAL = pd.concat([train_labels, val_labels], axis=0) # We only merge train and val as test will be the same for all splits

summary_dict = {"seed":[], "train":[], "val":[], "test":[], "train_ratio":[],"val_ratio":[],"test_ratio":[]}

for seed in seeds:

    train_copy = TRAIN_VAL.copy()
    splitter = GroupShuffleSplit(test_size=0.10, random_state = seed, n_splits=1)
    split_train_val = splitter.split(train_copy, groups=train_copy["patient_id"])
    train_idx, val_idx = next(split_train_val)
    new_train, new_val = train_copy.iloc[train_idx,:], train_copy.iloc[val_idx,:]

    train_ids = set(new_train.patient_id.unique())
    val_ids = set(new_val.patient_id.unique())

    if len(set.intersection(train_ids, val_ids)) == 0:
        print(" No overlap between patients in splits. Saving files...")
        # Eliminate the columns we created for the split because MedFuse code reads the whole row and not by csv columns
        # TODO: Improve this, there is a more elegant way to do this.
        added_cols = ["patient_id", "original_split"]
        original_cols = list(new_train.columns)
        cols_to_save = [col for col in original_cols if col not in added_cols]

        new_train.to_csv(f"laboratory/data/MedFuse/{task}/train_listfile_seed_{seed}_medfuse_test.csv", index=False, columns = cols_to_save)
        new_val.to_csv(f"laboratory/data/MedFuse/{task}/val_listfile_seed_{seed}_medfuse_test.csv", index=False, columns = cols_to_save)
    else:
        raise Exception("There is an overlap between patients in your splits.")
    
    print(f"Ratios seed {seed}")
    train_ratio = new_train.shape[0]/(new_train.shape[0]+new_val.shape[0]+test_labels.shape[0])
    val_ratio = new_val.shape[0]/(new_train.shape[0]+new_val.shape[0]+test_labels.shape[0])
    test_ratio = test_labels.shape[0]/(new_train.shape[0]+new_val.shape[0]+test_labels.shape[0])
    print(f" Train: {train_ratio}")
    print(f" Val: {val_ratio}")
    print(f" Test: {test_ratio}") # This should always be the same
    print(f" Saved train/val split files for seed {seed} on test type medfuse_test\n")
    summary_dict["seed"].append(seed)
    summary_dict["train"].append(new_train.shape[0])
    summary_dict["val"].append(new_val.shape[0])
    summary_dict["test"].append(test_labels.shape[0])
    summary_dict["train_ratio"].append(train_ratio)
    summary_dict["val_ratio"].append(val_ratio)
    summary_dict["test_ratio"].append(test_ratio)
print(f" Done creating split with args:\n{args}")
summary_df = pd.DataFrame.from_dict(summary_dict)
summary_df["total_elements"] = summary_df.apply(lambda x: x.train+x.val+x.test, axis=1)
print(summary_df)