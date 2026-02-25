import itertools
import random


method = "mfvi"

# parameters_mortality = {
#     # "learning_rate":[1e-5, 7.9439e-5, 8e-6],
#     "batch_size":[16],
#     "num_epochs":[5, 10, 15, 20, 30],
#     "reg_scale":[0, 0.1, 1, 10, 100],
#     "alpha": [0],
#     "prior_var": [1],
#     "mimic_task": ["in-hospital-mortality"],
#     "wandb_project": ["uq-fusion-det-mortality"]
#     }

# parameters_phenotyping = {
#     # "learning_rate":[1e-4, 1.8233e-4, 1.8233e-5],
#     "batch_size":[16],
#     "num_epochs":[5, 10, 15, 20, 30],
#     "reg_scale":[0, 0.1, 1, 10, 100],
#     "alpha": [0],
#     "prior_var": [1],
#     "mimic_task": ["phenotyping"],
#     "wandb_project": ["uq-fusion-det-phenotyping"]
#     }

# print(parameters_mortality)
# print()
# print(parameters_phenotyping)          
# keys, values = zip(*parameters_mortality.items())
# permutations_dicts_1 = [dict(zip(keys, v)) for v in itertools.product(*values)]

# keys, values = zip(*parameters_phenotyping.items())
# permutations_dicts_2 = [dict(zip(keys, v)) for v in itertools.product(*values)]


# for i, hyper_set in enumerate(permutations_dicts_1):
#     permutations_dicts_1[i]["learning_rate"] = random.uniform(10e-5, 10e-2)
#     permutations_dicts_2[i]["learning_rate"] = random.uniform(10e-5, 10e-2)

# print()
# print(permutations_dicts_1)
# print()
# print(permutations_dicts_2)
#-----------------------------------

# This code is to iterate over random samples of values
permutations_dicts_1 = []
permutations_dicts_2 = []

for i in range(100):
    # MORTALITY: BEST 0.0064659501030560436, 6.465e-3 
    sampled_lr = random.uniform(10e-5, 10e-2)
    permutations_dicts_1.append(
        {
            "batch_size":"16",
            "num_epochs":random.choice([5, 10, 15, 20, 25]),
            "learning_rate":random.choice([6.465e-2, 6.465e-3, 6.465e-4]),
            "alpha":"0",
            # "reg_scale":random.choice([0, 0.1, 1, 10, 100]),
            "prior_var":random.choice([0.1, 1, 10, 100, 1000]),
            "mimic_task": "in-hospital-mortality",
            "wandb_project": f"uq-fusion-{method}-mortality"
        }
    )
    # CLINICAL COND: BEST 0.000258107043673228, 2.581e-4
    # sampled_lr = random.uniform(10e-5, 10e-2)
    sampled_lr = random.uniform(1e-4, 3e-4)
    permutations_dicts_2.append(
        {
            "batch_size":"16",
            "num_epochs":random.choice([5, 10, 15, 20, 25]),
            "learning_rate":random.choice([2.581e-3, 2.581e-4, 2.581e-5]),
            "alpha":"0",
            # "reg_scale":random.choice([0, 0.1, 1, 10, 100]),
            "prior_var":random.choice([0.1, 1, 10, 100, 1000]),  
            "mimic_task": "phenotyping",
            "wandb_project": f"uq-fusion-{method}-phenotyping"
        }
    )


print(f"Total of {len(permutations_dicts_1)} hyperparams combinations for MORTALITY task.")
print(f"Total of {len(permutations_dicts_2)} hyperparams combinations for CLINICAL COND task.")
# print(f"CLINICAL COND, total of {len(permutations_dicts_phenotyping)} hyperparams combinations.")
# print(permutations_dicts)

#-----------------------------------
N = int(input("Input number of samples per task N: "))

selected_subset1 = random.sample(permutations_dicts_1, N)
selected_subset2 = random.sample(permutations_dicts_2, N)
selected_subset = selected_subset1 + selected_subset2
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
                    "checkpoints_best_models/7340086_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_300_seed_0_phenotyping_medfuse_test_40/checkpoint_40",
                    "checkpoints_best_models/4367437_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_301_seed_1_phenotyping_medfuse_test_40/checkpoint_40",
                    "checkpoints_best_models/9968083_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_302_seed_2_phenotyping_medfuse_test_40/checkpoint_40",
                    "checkpoints_best_models/5920095_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_303_seed_3_phenotyping_medfuse_test_40/checkpoint_40",
                    "checkpoints_best_models/0132415_Fusion_DET_40_0.0002581070436732287_16_0.0_1_exprt_304_seed_4_phenotyping_medfuse_test_40/checkpoint_40"
                    ]
}

seeds = [0, 1, 2, 3, 4]
experiments = []
for hyper_set in selected_subset:
    for seed in seeds:
        seed_hyperset = hyper_set.copy()
        seed_hyperset["seed"] = seed
        seed_hyperset["pretrained_prior_path"] = pre_trained_priors[seed_hyperset["mimic_task"]][seed]
        experiments.append(seed_hyperset)

print(f"Total number of experiments with {len(seeds)} seeds: {len(experiments)}")
with open(f"job_files/tuning/jobs_fusion_{method}_hypertuning.txt", "w") as f:

    for i, config in enumerate(experiments):
        k = list(config.keys())
        v = list(config.values())

        if method == "mfvi":
            base = f"python trainer.py --config configs/nn-tdvi-pt-fusion-mimic-{method}.json --jobid {i} --pretrained_prior --save_to_wandb --label_file_splits medfuse_test --model_for_final_eval LAST "
        else:
            base = f"python trainer.py --config configs/nn-tdvi-pt-fusion-mimic-{method}.json --jobid {i} --save_to_wandb --label_file_splits medfuse_test --model_for_final_eval LAST "
        # Iterate over each param to add it to the configuration
        hyperparams = ""
        for param in range(len(k)):
            hyperparams += f"--{k[param]} {v[param]} "
        hyperparams += "\n" # Jump line at the end of each experiment config
        final = base + hyperparams
        f.write(final)