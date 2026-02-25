import numpy as np
from sklearn.metrics import recall_score
import torch

task = "in-hospital-mortality"

for seed in [0,1,2,3,4]:
    print(f"Processing SEED: {seed}")
    for desired_recall in [0.7, 0.8, 0.9]:
        print(f" Processing RECALL: {desired_recall}")
        context_set = np.load(f"laboratory/train_points_for_context_{task}_seed_{seed}.npz")

        preds, targets = np.mean(context_set["preds"], axis=0), context_set["targets"]
        preds = torch.sigmoid(torch.from_numpy(preds))
        # preds.shape, targets.shape

        t = 0
        for t in np.linspace(1,0,100):
            recall = recall_score(targets , preds > t)
            if recall >= desired_recall:
                print(f"Final recall: {recall} | Final threshold: {t}")
                break
        
        context_preds = preds >  t

        misclassified_points = ~(context_preds == torch.from_numpy(targets))
        misclassified_points = misclassified_points.flatten()
        misclassified_points.sum()

        ehr_inputs = context_set["ehr_inputs"][misclassified_points]
        ehr_targets = context_set["ehr_targets"][misclassified_points]
        cxr_inputs = context_set["cxr_inputs"][misclassified_points]
        cxr_targets = context_set["cxr_targets"][misclassified_points]
        ehr_cxr_pairs = context_set["ehr_cxr_pairs"][misclassified_points]

        print(f"Shapes\n EHR inputs: {ehr_inputs.shape} | EHR targets: {ehr_targets.shape} | CXR inputs: {cxr_inputs.shape} | CXR targets: {cxr_targets.shape} | EHR-CXR pairs: {ehr_cxr_pairs.shape}")

        np.savez(f"laboratory/data/MedFuse/ContextPoints/Context-II/{task}/context_set_2_{task}_recall_{int(desired_recall*100)}_seed_{seed}.npz", 
                             ehr_inputs=ehr_inputs, 
                             ehr_targets=ehr_targets, 
                             cxr_inputs=cxr_inputs, 
                             cxr_targets=cxr_targets, 
                             ehr_cxr_pairs=ehr_cxr_pairs)
        
print(f"Finished SEED: {seed}")