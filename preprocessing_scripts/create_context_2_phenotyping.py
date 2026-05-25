import numpy as np
from sklearn.metrics import recall_score
import torch

task = "phenotyping"

for seed in [0,1,2,3,4]:
    print(f"Processing SEED: {seed}")
    for desired_recall in [0.7, 0.8, 0.9]:
        print(f" Processing RECALL: {desired_recall}")
        context_set = np.load(f"laboratory/train_points_for_context_{task}_seed_{seed}.npz")

        preds, targets = np.mean(context_set["preds"], axis=0), context_set["targets"]
        preds = torch.sigmoid(torch.from_numpy(preds))
        preds.shape, targets.shape

        context_preds = np.zeros(preds.shape)
        context_preds.shape

        context_preds = np.zeros(preds.shape)
        misclassified_points_labels = np.zeros(preds.shape)

        for label in range(preds.shape[1]):
            print(preds[:,:label+1].shape)
            s_preds = preds[:,label]
            s_targets = targets[:,label]
            
            for t in np.linspace(1,0,100):
                recall = recall_score(s_targets , s_preds > t)            
                if recall >= desired_recall:
                    print(f"Final recall: {recall} | Final threshold: {t}")
                    # print(t, recall)
                    break
            context_preds[:,label] = s_preds > t
            misclassified_points_labels[:,label] = (context_preds[:,label] == s_targets)

        misclassified_points_labels

        missclassified_points_t = {"t":[], "num_points":[]}
        correctly_classified_points_t = {"t":[], "num_points":[]}
        for t in range(1, 26):
            missclassified_points_t["t"].append(t)
            missclassified_points_t["num_points"].append((misclassified_points_labels.sum(axis=1) < t).sum())

            correctly_classified_points_t["t"].append(t)
            correctly_classified_points_t["num_points"].append((misclassified_points_labels.sum(axis=1) > t).sum())


        for final_t in [10, 15, 20]:
            # final_t = 20
            print(f" Processing FINAL_T: {final_t}")
            misclassified_points = (misclassified_points_labels.sum(axis=1) < final_t)
            # misclassified_points, misclassified_points.shape
            # misclassified_points.sum()

            ehr_inputs = context_set["ehr_inputs"][misclassified_points]
            ehr_targets = context_set["ehr_targets"][misclassified_points]
            cxr_inputs = context_set["cxr_inputs"][misclassified_points]
            cxr_targets = context_set["cxr_targets"][misclassified_points]
            ehr_cxr_pairs = context_set["ehr_cxr_pairs"][misclassified_points]

            print(f"Shapes\n EHR inputs: {ehr_inputs.shape} | EHR targets: {ehr_targets.shape} | CXR inputs: {cxr_inputs.shape} | CXR targets: {cxr_targets.shape} | EHR-CXR pairs: {ehr_cxr_pairs.shape}")

            np.savez(f"laboratory/data/MedFuse/ContextPoints/Context-II/{task}/context_set_2_{task}_recall_{int(desired_recall*100)}_correct_labels_{final_t}_seed_{seed}.npz", 
                                        ehr_inputs=ehr_inputs, 
                                        ehr_targets=ehr_targets, 
                                        cxr_inputs=cxr_inputs, 
                                        cxr_targets=cxr_targets, 
                                        ehr_cxr_pairs=ehr_cxr_pairs)
print(f"Finished SEED: {seed}")