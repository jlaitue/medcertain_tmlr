import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics
import pandas as pd
import jax
import jax.numpy as jnp

np.random.seed(42)

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
  y = np.argmax(y, axis=1)
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

def ece(targets, preds):
    labels = preds.shape[-1]
    ece_split_dict = {}
    total_ece_split = 0

    for label in range(labels):
        # We extract the preds/targets per label
        preds_test_label = preds[:, label]
        targets_test_label = targets[:, label]
        
        # Predictions
        # With this procedure we convert our preds into a one-hot encoded version that 
        # contains the predicted probability in the column index of the predicted class
        # Ex. [.21, .32, .61] --> [[.21, 0], [.32, 0], [0,.41]] for one-hot encoded targets: [[1,0], [1, 0], [0, 1]]
        one_hot_preds = (preds_test_label > 0.5).astype(float) 
        one_hot_preds = jax.nn.one_hot(one_hot_preds, num_classes=2)
        one_hot_preds_test_label = (one_hot_preds.T * preds_test_label).T
        
        # Targets
        # one-hote encoded targets --> [[1,0], [1, 0], [0, 1]]
        one_hot_targets_test_label = jax.nn.one_hot(targets_test_label, num_classes=2)
        
        ece_split_label = 100 * calibration(one_hot_targets_test_label, one_hot_preds_test_label)[0]

        ece_split_dict[f'label_{label+1}'] = ece_split_label
        total_ece_split += ece_split_label

    ece_split_dict['total_ece_split'] = total_ece_split/labels
    ece = ece_split_dict['total_ece_split']

    return ece

def selective_prediction_eval_protocol_1(targets, preds, eval_metric):  # Y is of shape (n, num_labels)

    '''Selective Prediction AUROC/AUPRC
    Evaluation protocol 1:
        1) Entropy for each point and we do this per label
        2) Sort the data points by level of entropy
        3) Threshold over points
        4) Calculate AUROC/AUPRC over non-excluded points
        5) Average AUROC/AUPRC per label
    Args:
        preds: (n, 25) n = batch size
        targets: (n, 25) n = batch size
        eval_metric: one of AUROC/AUPRC
    Returns:
        auc_sel_list: list of values per label for selective prediction AUROC/AUPRC
    '''

    discretization = 0  # if this doesn't work, try 2, 5, 10
    thresholds = np.linspace(start=100, stop=discretization, num=100)
    
    # preds = jnp.mean(jax.nn.sigmoid(logits), axis=0)

    # 1) Entropy for each point for each label
    all_scores = - (preds * jnp.log(preds + 1e-10))

    values_list = []# TODO shouldn't this be restarted eat each threshold loop?
    auc_sel_list = []
    
    for label in range(preds.shape[-1]):

        # Obtain the values per label: entropy scores, probabilities and targets
        scores = all_scores[:,label]
        label_preds = preds[:,label]
        label_targets = targets[:,label]

        label_thresholded_metric_values = []

        for threshold in thresholds:
            p = np.percentile(scores, threshold)

            # verify mask for mimic multilabel
            mask = np.array(scores <= p)

            preds_subset = label_preds[mask]  # all preds corresponding to non-rejected points (shape: (n_subset, num_labels))
            labels_subset = label_targets[mask]  # all labels corresponding to non-rejected points (shape: (n_subset, num_labels))

            try:
                if eval_metric == "AUROC":
                    label_metric = roc_auc_score(labels_subset, preds_subset, average=None)
                elif eval_metric == "AUPRC":
                    label_metric = average_precision_score(labels_subset, preds_subset, average=None)
                    # label_metric = average_precision_score(labels_subset, jax.nn.sigmoid(logits_subset), average=None)
                    # precision, recall, thresholds = precision_recall_curve(y_true, y_scores) # We can do this to verify the PRC curves are going
                else:
                    raise ValueError("Only AUROC or AUPRC are accepted evaluation metrics.")
            except:
                # If the threshold makes the auroc calculation fail,
                # the new auc would be at most as bad as the previous auc calculated?
                label_metric = sum(label_thresholded_metric_values) / len(label_thresholded_metric_values)

            label_thresholded_metric_values.append(label_metric)
        
        values = np.array(label_thresholded_metric_values)
        values_list.append(values)

        auc_sel = 0
        for i in range(len(thresholds)-1):
            if i == 0:
                x = 100 - thresholds[i+1]
            else:
                x = thresholds[i] - thresholds[i+1]
            auc_sel += (x * values[i] + x * values[i+1]) / 2
            
        auc_sel_list.append(auc_sel)

    auc_sel_value = np.array(auc_sel_list).mean()

    return auc_sel_value, auc_sel_list

def selective_prediction_eval_protocol_3(targets, preds, eval_metric, agg='mean'):  # Y is (n, num_labels)
    '''Selective Prediction AUROC/AUPRC
    Evaluation protocol 3:
        1) Entropy for each point and label
        2) We compute median or mean entropy across labels
        3) Sort the data points by level of entropy
        4) Threshold over points
        5) Calculate AUROC/AUPRC over non-excluded points
        Uses predictive entropy with T thresholds.
    Args:
        preds: (n, 25) n = batch size
        targets: (n, 25) n = batch size
        eval_metric: one of AUROC/AUPRC
    Returns:
        auc_sel: Selective Prediction AUROC/AUPRC using evaluation protocol 3
    '''
    discretization = 0  # if this doesn't work, try 2, 5, 10
    thresholds = np.linspace(start=100, stop=discretization, num=100)
    
    # 1) Entropy for each point each label
    all_scores = - (preds * np.log(preds + 1e-10))

    values_list = []

    # 2) Compute median or mean entropy across labels
    if agg == 'mean':
        scores = np.mean(all_scores, axis=-1)
    elif agg == 'median':
        scores = np.median(all_scores, axis=-1)

    thresholded_metric_values = []
    for threshold in thresholds:
        p = np.percentile(scores, threshold)
        mask = np.array(scores <= p)

        preds_subset = preds[mask]  # all preds corresponding to non-rejected points (shape: (n_subset, num_labels))
        labels_subset = targets[mask]  # all labels corresponding to non-rejected points (shape: (n_subset, num_labels))

        try:
            if eval_metric == "AUROC":
                metric = roc_auc_score(labels_subset, preds_subset, average=None)
            elif eval_metric == "AUPRC":
                metric = average_precision_score(labels_subset, preds_subset, average=None)
            else:
                raise ValueError("Only AUROC or AUPRC are accepted evaluation metrics.")
        except:
            # TODO this doesnt take into account if the first iteration fails, we would have division over 0!!
            # If the threshold makes the auroc calculation fail,
            # the new auroc would be at most as bad as the previous roc calculated
            metric = sum(thresholded_metric_values) / len(thresholded_metric_values)

        thresholded_metric_values.append(metric) # Each auroc is (25, )

    values = np.array(thresholded_metric_values)
    values_list.append(values)

    auc_sel = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]

        auc_sel += (x * values[i] + x * values[i+1]) / 2
    
    auc_sel_mean = np.array(auc_sel).mean()

    return auc_sel_mean, auc_sel

def multilabel_bce_with_sigmoid(targets, preds):
    '''BCE loss of multilabel distribution.
    '''
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    loss = -np.sum(targets*np.log(preds + 1e-10) + (1 - targets)*np.log(1 - preds + 1e-10), axis=0)
    loss = np.mean(loss)
    return loss

def evaluate_new(df, selected_metric):

    if selected_metric == "AUROC":
        metric_value = roc_auc_score(df['y_truth'], df['y_pred'], average="weighted")

    elif selected_metric == "AUPRC":
        metric_value = average_precision_score(df['y_truth'], df['y_pred'], average="weighted")

    elif selected_metric == "SEL_AUROC_1":
        metric_value,  _ = selective_prediction_eval_protocol_1(targets=df['y_truth'], preds=df['y_pred'], eval_metric="AUROC")
    
    elif selected_metric == "SEL_AUPRC_1":
        metric_value, _ = selective_prediction_eval_protocol_1(targets=df['y_truth'], preds=df['y_pred'], eval_metric="AUPRC")

    elif selected_metric == "SEL_AUROC_3":
        metric_value, _ = selective_prediction_eval_protocol_3(targets=df['y_truth'], preds=df['y_pred'], eval_metric="AUROC")
    
    elif selected_metric == "SEL_AUPRC_3":
        metric_value, _ = selective_prediction_eval_protocol_3(targets=df['y_truth'], preds=df['y_pred'], eval_metric="AUPRC")

    elif selected_metric == "ECE":
        metric_value = ece(targets=df['y_truth'], preds=df['y_pred'])
    
    elif selected_metric == "NLL":
        metric_value = multilabel_bce_with_sigmoid(targets=df['y_truth'], preds=df['y_pred']) #Check the aggregation

    return metric_value

def bootstraping_eval(df, selected_metric, num_iter):
    """This function samples from the testing dataset to generate a list of performance metrics using bootstraping method"""
    metric_values_list = []

    for _ in range(num_iter):
        y_truth = df["y_truth"]
        y_pred = df["y_pred"]
        idx = np.random.choice(np.arange(y_truth.shape[0]), y_truth.shape[0], replace=True)
        y_truth, y_pred = y_truth[idx], y_pred[idx]
        sample = {'y_truth': y_truth, 'y_pred': y_pred}

        # sample = df.sample(frac=1, replace=True)

        metric_value = evaluate_new(sample, selected_metric)

        metric_values_list.append(metric_value)
    return metric_values_list

def computing_confidence_intervals(list_, true_value):
    """This function calculates the 95% Confidence Intervals"""

    delta = (true_value - list_)
    delta_lower = np.percentile(delta, 97.5, axis=0)
    delta_upper = np.percentile(delta, 2.5, axis=0)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    
    return (upper, lower)

def get_model_performance(df, selected_metric):

    num_iter = 1000

    test_metric_value = evaluate_new(df, selected_metric) # Obtain the base metric value

    metric_values_list = bootstraping_eval(df, selected_metric, num_iter) # Get list of values for the metrics

    upper_metric_value, lower_metric_value = computing_confidence_intervals(metric_values_list, test_metric_value)# Compute CI 

    print(f"\nSAMPLES for {selected_metric}:\n{metric_values_list}\n")

    return (test_metric_value, upper_metric_value, lower_metric_value) # Return the values

def compute_metric_and_ci(y_true, predictions, selected_metric="AUROC"):
        y_true = np.array(y_true)
        predictions = np.array(predictions)
        
        # df = pd.DataFrame({'y_truth': y_true, 'y_pred': predictions})
        df = {'y_truth': y_true, 'y_pred': predictions}
        (metric_value, upper_metric_value, lower_metric_value) = get_model_performance(df, selected_metric)

        ci_metric = (lower_metric_value, upper_metric_value)

        return metric_value, ci_metric