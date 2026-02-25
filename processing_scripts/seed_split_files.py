import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--task", default="in-hospital-mortality", help="MIMIC task. One of 'in-hospital-mortality' or 'phenotyping'")
parser.add_argument("-st", "--test_split_type", default="same", help="Type of test split. Use 'same' or 'seed'")
parser.add_argument("-s", "--n_seeds", default=5, type=int, help="Number of seeds to run")
args = vars(parser.parse_args())
 
# Set up parameters
task = args["task"]
test_split_type = args["test_split_type"]
n_seeds = args["n_seeds"]
seeds = list(range(0, n_seeds))
mimic_ehr_path = 'laboratory/data/mimic-iv-extracted/' # MIMIC EHR dataset location

print(f"Current settings for creation of splits\n{args}\n")
input("Press any key to continue...")

# Load original files from host of MIMIC dataset
train_labels = pd.read_csv(f"{mimic_ehr_path}/{task}/train_listfile.csv")
val_labels = pd.read_csv(f"{mimic_ehr_path}/{task}/val_listfile.csv")
test_labels = pd.read_csv(f"{mimic_ehr_path}/{task}/test_listfile.csv")

# We add a new column patient_id which is based on the filename column of the dfs
train_labels["patient_id"] = train_labels.apply(lambda x: x.stay.split("_")[0], axis=1)
val_labels["patient_id"] = val_labels.apply(lambda x: x.stay.split("_")[0], axis=1)
test_labels["patient_id"] = test_labels.apply(lambda x: x.stay.split("_")[0], axis=1)
train_labels["original_split"], val_labels["original_split"], test_labels["original_split"] = "train", "val", "test"
full_df = pd.concat([train_labels, val_labels, test_labels], axis=0)

for selected_test_seed in range(10):
    print(f"Running fixed test set with: {selected_test_seed}")
    if test_split_type == "same":
        # selected_test_seed = 0

        splitter_test = GroupShuffleSplit(test_size=0.20, random_state = selected_test_seed, n_splits=1)
        split_train_test = splitter_test.split(full_df, groups=full_df["patient_id"])
        train_idx, test_idx = next(split_train_test)
        
        NO_TEST, TEST = full_df.iloc[train_idx,:], full_df.iloc[test_idx,:]

    for seed in seeds:

        if test_split_type == "seed":
            splitter_test = GroupShuffleSplit(test_size=0.20, random_state = seed, n_splits=1)
            split_train_test = splitter_test.split(full_df, groups=full_df["patient_id"])
            train_idx, test_idx = next(split_train_test)
            NO_TEST, TEST = full_df.iloc[train_idx,:], full_df.iloc[test_idx,:]

        new_train = NO_TEST.copy()
        splitter_val = GroupShuffleSplit(test_size=0.10, random_state = seed, n_splits=1)
        split_train_val = splitter_val.split(new_train, groups=new_train["patient_id"])
        train_idx, val_idx = next(split_train_val)
        new_train, new_val = new_train.iloc[train_idx,:], new_train.iloc[val_idx,:]

        train_ids = set(new_train.patient_id.unique())
        val_ids = set(new_val.patient_id.unique())
        test_ids = set(TEST.patient_id.unique())

        if len(set.intersection(train_ids, val_ids, test_ids)) == 0:
            print(" No overlap between patients in splits. Saving files...")
            # Eliminate the columns we created for the split because MedFuse code reads the whole row and not by columns
            # TODO: Improve this, there is a more elegant way to do this.
            added_cols = ["patient_id", "original_split"]
            original_cols = list(new_train.columns)
            saved_cols = [col for col in original_cols if col not in added_cols]

            new_train.to_csv(f"laboratory/data/{task}/train_listfile_seed_{seed}_{test_split_type}_test_{selected_test_seed}.csv", index=False, columns = saved_cols)
            new_val.to_csv(f"laboratory/data/{task}/val_listfile_seed_{seed}_{test_split_type}_test_{selected_test_seed}.csv", index=False, columns = saved_cols)

            if test_split_type == "seed":
                new_train.to_csv(f"laboratory/data/{task}/train_listfile_seed_{seed}_{test_split_type}_test.csv", index=False, columns = saved_cols)
                new_val.to_csv(f"laboratory/data/{task}/val_listfile_seed_{seed}_{test_split_type}_test.csv", index=False, columns = saved_cols)
                TEST.to_csv(f"laboratory/data/{task}/test_listfile_seed_{seed}_{test_split_type}_test.csv", index=False, columns = saved_cols)
            else:
                new_train.to_csv(f"laboratory/data/{task}/train_listfile_seed_{seed}_{test_split_type}_test_{selected_test_seed}.csv", index=False, columns = saved_cols)
                new_val.to_csv(f"laboratory/data/{task}/val_listfile_seed_{seed}_{test_split_type}_test_{selected_test_seed}.csv", index=False, columns = saved_cols)
                TEST.to_csv(f"laboratory/data/{task}/test_listfile_{test_split_type}_test_{selected_test_seed}.csv", index=False, columns = saved_cols)
        else:
            raise Exception("There is an overlap between patients in your splits.")
        print("Ratios")
        print(f" Train: {new_train.shape[0]/(new_train.shape[0]+new_val.shape[0]+TEST.shape[0])}")
        print(f" Val: {new_val.shape[0]/(new_train.shape[0]+new_val.shape[0]+TEST.shape[0])}")
        print(f" Test: {TEST.shape[0]/(new_train.shape[0]+new_val.shape[0]+TEST.shape[0])}")
        print(f" Saved train/val split files based for seed {seed} on test type {test_split_type}")
        print(f"  {test_split_type}_test_{selected_test_seed}.csv")
    print(f" Done creating split with args:\n{args}")
    print("\n\n")
