import itertools
import random


method = "psvi"

# parameters = {
#     "prior_likelihood_scale":[0.1, 1, 10], 
#     "prior_likelihood_f_scale":[0, 1, 10],
#     "prior_likelihood_cov_scale":[0.1, 0.01, 0.001], 
#     "prior_likelihood_cov_diag":[0.5, 1, 5]
#     }

parameters_mortality = {
    "learning_rate":[6.465e-2, 6.465e-3, 6.465e-4],
    "batch_size":[16],
    "num_epochs":[5, 10, 15, 20, 25],
    "alpha": [0],
    # ---------------------
    # Parameters for method
    "context_batch_size":[16, 32],
    "context_points":["train"],
    "prior_var": [0.1, 1, 10, 1000],
    "prior_likelihood_scale":[0.1, 1, 10], 
    "prior_likelihood_f_scale":[0, 1, 10],
    "prior_likelihood_cov_scale":[0.1, 0.01, 0.001], 
    "prior_likelihood_cov_diag":[0.5, 1, 5],
    # ---------------------
    "mimic_task": ["in-hospital-mortality"],
    "wandb_project": [f"uq-fusion-{method}-mortality"]
    }

parameters_phenotyping = {
    "learning_rate":[2.581e-3, 2.581e-4, 2.581e-5],
    "batch_size":[16],
    "num_epochs":[5, 10, 15, 20, 25],
    "alpha": [0],
    # ---------------------
    # Parameters for method
    "context_batch_size":[16, 32],
    "context_points":["merged"],
    "prior_var": [0.1, 1, 10, 100, 1000],
    "prior_likelihood_scale":[0.1, 1, 10], 
    "prior_likelihood_f_scale":[0, 1, 10],
    "prior_likelihood_cov_scale":[0.1, 0.01, 0.001], 
    "prior_likelihood_cov_diag":[0.5, 1, 5],
    "reg_scale":[1, 0.1, 0.01],
    "reg_scale_gap":[1, 0.1, 0.01],
    "rho_sam": [0.1, 0.01, 0.001, 0.0001],

    # ---------------------
    "mimic_task": ["phenotyping"],
    "wandb_project": [f"uq-gap-finetuning"]
    }

# print(parameters_mortality)
# print()
# print(parameters_phenotyping)          
keys, values = zip(*parameters_mortality.items())
permutations_dicts_1 = [dict(zip(keys, v)) for v in itertools.product(*values)]

keys, values = zip(*parameters_phenotyping.items())
permutations_dicts_2 = [dict(zip(keys, v)) for v in itertools.product(*values)]


#-----------------------------------

# This code is to iterate over random samples of values
# permutations_dicts_1 = []
# permutations_dicts_2 = []

# for i in range(100):
#     # MORTALITY = lr:[6e-5, 9e-5], alpha:[0.5, 0.9]

#     sampled_lr = random.uniform(6e-5, 9e-5)
#     permutations_dicts_1.append(
#         {
#             # "learning_rate":f"{sampled_lr}"[:6]+"e-5",
#             "learning_rate":"7.9439e-5",
#             "alpha":"0.6571",
#             "prior_var":f"{random.uniform(0.1, 10)}",
#             "num_epochs": 100,
#             "mimic_task": "in-hospital-mortality",
#             "wandb_project": "uq-fusion-mfvi-mortality"
#         }
#     )
#     # CLINICAL COND = lr:[1e-4, 3e-4], alpha:[0.01, 0.9]
#     sampled_lr = random.uniform(1e-4, 3e-4)
#     permutations_dicts_2.append(
#         {
#             # "learning_rate":f"{sampled_lr/100}"[:6]+"e-4",
#             "learning_rate":"1.8233e-4",
#             "alpha":"0.6343",
#             "prior_var":f"{random.uniform(0.1, 10)}",
#             "num_epochs": 100,
#             "mimic_task": "phenotyping",
#             "wandb_project": "uq-fusion-mfvi-phenotyping"
#         }
#     )


# print(f"Total of {len(permutations_dicts_1)} hyperparams combinations for MORTALITY task.")
# print(f"Total of {len(permutations_dicts_2)} hyperparams combinations for CLINICAL COND task.")
# print(f"CLINICAL COND, total of {len(permutations_dicts_phenotyping)} hyperparams combinations.")
# print(permutations_dicts)

#-----------------------------------
N = int(input("Input number of samples per task N: "))

selected_subset1 = random.sample(permutations_dicts_1, N)
selected_subset2 = random.sample(permutations_dicts_2, N)
# selected_subset = selected_subset1 + selected_subset2
selected_subset = selected_subset2
random.shuffle(selected_subset)
print(f"Selected sample size: {len(selected_subset)}")

print()

pre_trained_priors = {
    "in-hospital-mortality":
                    [
                    "checkpoints_best_models/3535469_Fusion_DET_20_0.0064659501030560436_16_0.0_1_exprt_390_seed_0_in-hospital-mortality_medfuse_test_20/checkpoint_20",
                    "checkpoints_best_models/7869692_Fusion_DET_20_0.0064659501030560436_16_0.0_1_exprt_391_seed_1_in-hospital-mortality_medfuse_test_20/checkpoint_20",
                    "checkpoints_best_models/7729893_Fusion_DET_20_0.0064659501030560436_16_0.0_1_exprt_392_seed_2_in-hospital-mortality_medfuse_test_20/checkpoint_20",
                    "checkpoints_best_models/6379693_Fusion_DET_20_0.0064659501030560436_16_0.0_1_exprt_393_seed_3_in-hospital-mortality_medfuse_test_20/checkpoint_20",
                    "checkpoints_best_models/929203_Fusion_DET_20_0.0064659501030560436_16_0.0_1_exprt_394_seed_4_in-hospital-mortality_medfuse_test_20/checkpoint_20"
                    ],
    "phenotyping":
                    [
                    "all_best_models/checkpoints_to_select_best_07_24/368213_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_300_seed_0_phenotyping_medfuse_test_full_training_40/checkpoint_40",
                    "all_best_models/checkpoints_to_select_best_07_24/6638105_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_301_seed_1_phenotyping_medfuse_test_full_training_40/checkpoint_40",
                    "all_best_models/checkpoints_to_select_best_07_24/6601634_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_302_seed_2_phenotyping_medfuse_test_full_training_40/checkpoint_40",
                    "all_best_models/checkpoints_to_select_best_07_24/5890853_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_303_seed_3_phenotyping_medfuse_test_full_training_40/checkpoint_40",
                    "all_best_models/checkpoints_to_select_best_07_24/7011003_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_304_seed_4_phenotyping_medfuse_test_full_training_40/checkpoint_40"
                    ]
}

seeds = [0, 1, 2, 3, 4]
experiments = []
for hyper_set in selected_subset:
    for seed in seeds:
        seed_hyperset = hyper_set.copy()
        seed_hyperset["pretrained_prior_path"] = pre_trained_priors[seed_hyperset["mimic_task"]][seed]
        seed_hyperset["context_points_data_file"] = "Context-III/phenotyping/context_set_3_cos_sim_type1_phenotyping_1_std.npz"
        seed_hyperset["seed"] = seed
        experiments.append(seed_hyperset)

print(f"Total number of experiments with {len(seeds)} seeds: {len(experiments)}")
with open(f"job_files/gap/gap_medcertain_finetuning_2.txt", "w") as f:

    for i, config in enumerate(experiments):
        k = list(config.keys())
        v = list(config.values())

        base = f"python trainer.py --config configs/nn-tdvi-pt-fusion-mimic-{method}.json --jobid {i} --pretrained_prior --gap_regularizer --save_to_wandb --label_file_splits medfuse_test --model_for_final_eval LAST "
        # Iterate over each param to add it to the configuration
        hyperparams = ""
        for param in range(len(k)):
            hyperparams += f"--{k[param]} {v[param]} "
        hyperparams += "\n" # Jump line at the end of each experiment config
        final = base + hyperparams
        f.write(final)