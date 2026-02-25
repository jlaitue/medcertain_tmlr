import numpy as np
from sklearn.metrics import recall_score
import torch
from sklearn.metrics.pairwise import cosine_similarity

for task in ["in-hospital-mortality", "phenotyping"]:
    # task = "in-hospital-mortality"
    latent_points = np.load(f"laboratory/latent_space_representation_{task}.npz")

    for seed in [0,1,2,3,4]:
        original_train_points = np.load(f"laboratory/train_points_for_context_{task}_seed_{seed}.npz")

        latent_ehr_points = latent_points["ehr_features"]
        latent_cxr_points = latent_points["cxr_features"]
        
        # Inter modality similarity
        cosine_distances = np.diag(cosine_similarity(latent_ehr_points, latent_cxr_points))
        mean_distance = cosine_distances.mean()
        std_distance = cosine_distances.std()

        # Intra modality similarity
        mean_distance_ehr = latent_ehr_points.mean()
        mean_distance_cxr = latent_cxr_points.mean()

        cosine_distances_ehr = np.diag(cosine_similarity(latent_ehr_points, mean_distance_ehr))
        cosine_distances_cxr = np.diag(cosine_similarity(latent_ehr_points, mean_distance_cxr))
        
        std_distance = cosine_distances.std()

        for n_std in [1,2]:
            # n_std = 1
            left_tail = mean_distance - n_std*std_distance
            context_points_idx = (cosine_distances < left_tail)

            print(f"Misclassified points: {context_points_idx.sum()} | Mean: {mean_distance:0.6f} | STD: {std_distance:0.6f} | Left tail: {left_tail:0.6f}")

            ehr_inputs = original_train_points["ehr_inputs"][context_points_idx]
            ehr_targets = original_train_points["ehr_targets"][context_points_idx]
            cxr_inputs = original_train_points["cxr_inputs"][context_points_idx]
            cxr_targets = original_train_points["cxr_targets"][context_points_idx]
            ehr_cxr_pairs = original_train_points["ehr_cxr_pairs"][context_points_idx]

            print(f"Shapes\n EHR inputs: {ehr_inputs.shape} | EHR targets: {ehr_targets.shape} | CXR inputs: {cxr_inputs.shape} | CXR targets: {cxr_targets.shape} | EHR-CXR pairs: {ehr_cxr_pairs.shape}")

            np.savez(f"laboratory/data/ContextPoints/Context-III/{task}/context_set_3_cos_sim_type1_{task}_{n_std}_std_seed_{seed}.npz", 
                                        ehr_inputs=ehr_inputs, 
                                        ehr_targets=ehr_targets, 
                                        cxr_inputs=cxr_inputs, 
                                        cxr_targets=cxr_targets, 
                                        ehr_cxr_pairs=ehr_cxr_pairs)