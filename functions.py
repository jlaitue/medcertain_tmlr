import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score, precision_recall_curve
import pandas as pd
import warnings
warnings.filterwarnings("error")

EPSILON = 1e-6

LABEL_TITLES = {
    0 : "Acute and unspecified renal failure",
    1 : "Acute cerebrovascular disease",
    2 : "Acute myocardial infarction",
    3 : "Cardiac dysrhythmias",
    4 : "Chronic kidney disease",
    5 : "Chronic obstructive pulmonary disease",
    6 : "Complications of surgical procedures",
    7 : "Conduction disorders",
    8 : "Congestive heart failure; nonhypertensive",
    9 : "Coronary atherosclerosis and other heart disease",
    10 : "Diabetes mellitus with complications",
    11 : "Diabetes mellitus without complication",
    12 : "Disorders of lipid metabolism",
    13 : "Essential hypertension",
    14 : "Fluid and electrolyte disorders",
    15 : "Gastrointestinal hemorrhage",
    16 : "Hypertension with complications",
    17 : "Other liver diseases",
    18 : "Other lower respiratory disease",
    19 : "Other upper respiratory disease",
    20 : "Pleurisy; pneumothorax",
    21 : "Pneumonia (except caused by tuberculosis or std)",
    22 : "Respiratory failure",
    23 : "Septicemia (except in labor)",
    24 : "Shock"
}

def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.
  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263
  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins
  Returns:
    ece: Expected Calibration Error
    mce: Maximum Calibration Error
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
#   y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]

  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  return ece, mce

@jax.jit
def accuracy(logits_or_p, Y):
    '''Compute accuracy
    Arguments:
        logits_or_p: (B, d)
        Y: (B,) integer labels.
    '''
    if len(Y) == 0:
        return 0.
    matches = jnp.argmax(logits_or_p, axis=-1) == Y
    return jnp.mean(matches)

@jax.jit
def categorical_nll(logits, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return optax.softmax_cross_entropy_with_integer_labels(logits, Y)

@jax.jit
def categorical_nll_with_softmax(p, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return -jnp.sum(jnp.log(p + 1e-10) * jax.nn.one_hot(Y, p.shape[-1]), axis=-1)

@jax.jit
def gaussian_nll(f, Y, likelihood_var):
    '''Negative log-likelihood of Gaussian distribution.
    '''
    likelihood = tfd.Normal(f, likelihood_var ** 0.5)
    nll = jnp.sum(-likelihood.log_prob(Y), -1)
    return nll

@jax.jit
def mimic_ce_loss_sigmoid(preds, Y): #shape (batch_size, labels)
    '''Cross-Entropy loss of multilabel/binary distribution for MIMIC.
    Input
    preds: Predictions already passed through sigmoid function
    Y: Ground truth labels

    Returns:
    Cross-Entropy loss for either binary or multilabel cases.
    '''
    return -jnp.sum(Y*jnp.log(preds + 1e-10) + (1 - Y)*jnp.log(1 - preds + 1e-10), axis=0)

@jax.jit
def categorical_entropy(p):
    '''Entropy of categorical distribution.
    Arguments:
        p: (B, d)

    Returns:
        (B,)
    '''
    return -jnp.sum(p * jnp.log(p + EPSILON), axis=-1)

# @jax.jit
def selective_accuracy(p, Y):
    '''Selective Prediction Accuracy
    Uses predictive entropy with T thresholds.
    Arguments:
        p: (B, d)

    Returns:
        (B,)
    '''

    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    # predictions_test = p.argmax(-1) # Removed from original code
    predictions_test = p
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = np.percentile(scores_id, threshold)
        mask = np.array(scores_id <= p)
        thresholded_accuracies.append(np.mean(accuracies_test[mask]))
    values_id = np.array(thresholded_accuracies)

    auc_sel_id = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel_id += (x * values_id[i] + x * values_id[i+1]) / 2

    return auc_sel_id

def fpr_tpr_scores(targets, preds, task):
    num_labels = 1
    if task == "phenotyping":
        num_labels = 25

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for label in range(num_labels):
        fpr[label], tpr[label], _ = roc_curve(targets[:, label], preds[:, label])
        roc_auc[label] = auc(fpr[label], tpr[label])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(targets.ravel(), preds.ravel())
    roc_auc["micro"] = np.array(auc(fpr["micro"], tpr["micro"]))

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[label] for label in range(num_labels)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for label in range(num_labels):
        mean_tpr += np.interp(all_fpr, fpr[label], tpr[label])

    # Finally average it and compute AUC
    mean_tpr /= num_labels

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = np.array(auc(fpr["macro"], tpr["macro"]))

    return fpr, tpr, roc_auc

def precision_recall_scores(targets, preds, task):
    num_labels = 1
    if task == "phenotyping":
        num_labels = 25

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    for label in range(num_labels):
        precision[label], recall[label], _ = precision_recall_curve(targets[:, label], preds[:, label])
        average_precision[label] = average_precision_score(targets[:, label], preds[:, label])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(targets.ravel(), preds.ravel())

    average_precision["micro"] = np.array(average_precision_score(targets, preds, average="micro"))

    return precision, recall, average_precision

def selective_prediction_mimic(targets, preds, eval_metric):  # Y is of shape (n, num_labels)

    '''Selective Prediction AUROC/AUPRC
    Evaluation protocol MIMIC:
        1) Entropy for each point and we do this per label
        2) Sort the data points by level of entropy
        3) Threshold over points
        4) Calculate AUROC/AUPRC over non-excluded points
        5) Average AUROC/AUPRC per label
    Args:
        preds: (n, z) n = batch size, z = labels
        targets: (n, z) n = batch size,  z = labels
        eval_metric: one of "AUROC" or "AUPRC"
    Returns:
        auc_sel_mean: average selective auc across all labels and thresholds
        auc_sel_labels: average selective auc across thresholds for each invidual label 
    '''
    # warnings.filterwarnings("error") # This should be use for our try except clauses
    discretization = 0  # if this doesn't work, try 2, 5, 10
    thresholds = np.linspace(start=100, stop=discretization, num=100)
    
    # 1) Entropy for each point for each label
    all_scores = - (preds * jnp.log(preds + EPSILON))

    num_labels = preds.shape[1]
    n_thresholds = thresholds.shape[0]
    labels_selective_auc = np.ones((num_labels, n_thresholds))

    print(f"    Processing METRIC {eval_metric}")
    for label in range(num_labels):
        print(f"     Processing label {label+1}: {LABEL_TITLES[label]}")
        # Obtain the values per label: entropy scores, probabilities and targets
        scores = all_scores[:,label]

        label_preds = preds[:,label]
        label_targets = targets[:,label]

        auc_array = np.ones(n_thresholds,)

        for idx_threshold, threshold in enumerate(thresholds):
            p = np.percentile(scores, threshold)

            # verify mask for mimic multilabel
            mask = np.array(scores <= p)

            preds_subset = label_preds[mask]  # all preds corresponding to non-rejected points (shape: (n_subset, num_labels))
            targets_subset = label_targets[mask]  # all labels corresponding to non-rejected points (shape: (n_subset, num_labels))

            try:
                if eval_metric == "AUROC":
                    label_auc = roc_auc_score(targets_subset, preds_subset, average=None)
                elif eval_metric == "AUPRC":
                    label_auc = average_precision_score(targets_subset, preds_subset, average=None)
                else:
                    raise ValueError("Only AUROC or AUPRC are accepted evaluation metrics.")
            except Exception as e:
                # If the threshold makes the auc calculation fail, we assign the value to the mean of the previous
                # compute selective auc
                # print(f"    {e} Label {label} threshold {threshold} {eval_metric}")
                label_auc = auc_array.mean()
            
            auc_array[idx_threshold] = label_auc
        
        labels_selective_auc[label, :] = auc_array

    auc_sel_mean = labels_selective_auc.mean()
    auc_sel_labels = labels_selective_auc.mean(axis=1)

    return auc_sel_mean, auc_sel_labels

def selective_accuracy_test_ood(p_id, p_ood, Y):
    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    predictions_test = p_id.argmax(-1)
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p_id)

    accuracies_ood = jnp.zeros(p_ood.shape[0])
    scores_ood = categorical_entropy(p_ood)

    accuracies = jnp.concatenate([accuracies_test, accuracies_ood], axis=0)
    scores = jnp.concatenate([scores_id, scores_ood], axis=0)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = np.percentile(scores, threshold)
        mask = np.array(scores <= p)
        thresholded_accuracies.append(np.mean(accuracies[mask]))
    values = np.array(thresholded_accuracies)

    auc_sel = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel += (x * values[i] + x * values[i+1]) / 2

    return auc_sel

def auroc_logits(predicted_logits_test, predicted_logits_ood, score, rng_key):
    predicted_targets_test = jax.nn.softmax(predicted_logits_test, axis=-1)
    predicted_targets_ood = jax.nn.softmax(predicted_logits_ood, axis=-1)

    ood_size = predicted_targets_ood.shape[1]
    test_size = predicted_targets_test.shape[1]
    anomaly_targets = jnp.concatenate((np.zeros(test_size), np.ones(ood_size)))
    if score == "entropy":
        entropy_test = categorical_entropy(predicted_targets_test.mean(0))
        entropy_ood = categorical_entropy(predicted_targets_ood.mean(0))
        scores = jnp.concatenate((entropy_test, entropy_ood))
    if score == "expected entropy":
        entropy_test = categorical_entropy(predicted_targets_test).mean(0)
        entropy_ood = categorical_entropy(predicted_targets_ood).mean(0)
        scores = jnp.concatenate((entropy_test, entropy_ood))
    elif score == "mutual information":
        mutual_information_test = categorical_entropy(predicted_targets_test.mean(0)) - categorical_entropy(predicted_targets_test).mean(0)
        mutual_information_ood = categorical_entropy(predicted_targets_ood.mean(0)) - categorical_entropy(predicted_targets_ood).mean(0)
        scores = jnp.concatenate((mutual_information_test, mutual_information_ood))
    else:
        NotImplementedError
    fpr, tpr, _ = roc_curve(anomaly_targets, scores)
    auroc_score = roc_auc_score(anomaly_targets, scores)
    return auroc_score

def merge_params(params_1, params_2):
    flat_params_1 = flax.traverse_util.flatten_dict(params_1)
    flat_params_2 = flax.traverse_util.flatten_dict(params_2)
    flat_params = flat_params_1 | flat_params_2
    unflat_params = flax.traverse_util.unflatten_dict(flat_params)
    return unflat_params

def split_params(params, type="dense"):
    flat_params_fixed = flax.traverse_util.flatten_dict(params)
    flat_params_rest = flax.traverse_util.flatten_dict(params)
    keys = flat_params_fixed.keys()

    i = -1
    for key in list(keys):
        if "Dense" in str(key) and "kernel" in str(key):
            i += 1

    if type == "dense":
        for key in list(keys):
            if f"Dense_{i}" in str(key):  # first check if there may be two final dense layers
                flat_params_fixed.pop(key)
            else:
                flat_params_rest.pop(key)
    elif type == "batch_norm":
        for key in list(keys):
            if "BatchNorm" in str(key):
                flat_params_fixed.pop(key)
            else:
                flat_params_rest.pop(key)
    else:
        raise NotImplementedError
        
    unflat_params_fixed = flax.traverse_util.unflatten_dict(flat_params_fixed)
    unflat_params_fixed = unflat_params_fixed
    unflat_params_rest = flax.traverse_util.unflatten_dict(flat_params_rest)
    unflat_params_rest = unflat_params_rest

    return unflat_params_fixed, unflat_params_rest

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
    """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.
    Args:
        y: A pytree of numpy ndarray, vector y in the equation above.
    """
    vector_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
    normalized_vector = jax.tree_map(lambda x: x / vector_norm, y)
    return normalized_vector