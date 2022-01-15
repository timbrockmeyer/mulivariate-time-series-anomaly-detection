from .metrics import precision_recall, F_score
from . import utils
import numpy as np

def evaluate_performance(train_res, test_res, threshold_method='max', smoothing=4, smoothing_method='mean'):
    '''
    Returns precision, recall, f1 and f2 scores.
    Determines anomaly threshold from normalized and smoothed validation data.
    Normalization is performed on each 1d sensor time series.
    Anomaly predictions are calculated as smoothed and normalized test error scores that exceed the threshold.

    Args:
        train_res (list): List of length three holding prediction and groundtruth values from validation. Third
                        Entry is assumed to be NoneType for nonexisting anomaly labels.
        test_res (list): List of length three holding prediction, groundtruth values and anomaly labels testing.
    '''
    train_pred_err, _ = train_res
    test_pred_err, anomaly_labels = test_res

    assert test_pred_err.size(1) == anomaly_labels.size(0)

    # row-wise normalization (within each sensor) and subsequent 1d smoothing
    train_error = utils.normalize_with_median_iqr(train_pred_err)
    test_error = utils.normalize_with_median_iqr(test_pred_err)
    if smoothing > 0:
        train_error = utils.weighted_average_smoothing(train_error, k=smoothing, mode=smoothing_method)
        test_error = utils.weighted_average_smoothing(test_error, k=smoothing, mode=smoothing_method)
    
    if threshold_method == 'max':
        anomaly_predictions = _max_thresholding(train_error, test_error)
    elif threshold_method == 'mean':
        anomaly_predictions = _mean_thresholding(train_error, test_error)
    elif threshold_method == 'best':
        anomaly_predictions, threshold_method = _best_thresholding(test_error, anomaly_labels)

    # evaluate test performance
    assert anomaly_predictions.shape == anomaly_labels.shape
    precision, recall = precision_recall(anomaly_predictions, anomaly_labels)
    f1 = F_score(precision, recall, beta=1)
    f2 = F_score(precision, recall, beta=2)

    # adjusted performance
    adjusted_predictions, latency = adjust_predicts(anomaly_predictions, anomaly_labels, calc_latency=True)
    precision_adj, recall_adj = precision_recall(adjusted_predictions, anomaly_labels)
    f1_adj = F_score(precision_adj, recall_adj, beta=1)
    f2_adj = F_score(precision_adj, recall_adj, beta=2)

    results_dict = {
        'method': threshold_method,
        'prec': precision,
        'rec': recall,
        'f1': f1,
        'f2': f2,
        'a_prec': precision_adj,
        'a_rec': recall_adj,
        'a_f1': f1_adj,
        'a_f2': f2_adj,
        'latency': latency
    }
    return results_dict

def adjust_predicts(pred, label, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels

    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    anomaly_state = False
    anomaly_count = 0 # number of anomalies found
    latency = 0

    for i in range(len(pred)):
        if label[i] and pred[i] and not anomaly_state: # if correctly found anomaly 
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1): # go backward until beginning of anomaly
                if not label[j]: # BEGINNING of anomaly
                    break
                else:
                    if not pred[j]: # set prediction to true
                        pred[j] = True
                        latency += 1
        elif not label[i]: # END of anomaly
            anomaly_state = False
        if anomaly_state: # still in anomaly and was already found
            pred[i] = True
    if calc_latency:
        return pred, latency / (anomaly_count + 1e-8)
    else:
        return pred

def _max_thresholding(train_errors, test_errors):
    '''
    Returns anomaly predictions on test errors based on threshold
    calculated on the validation errors. 
    Threshold is the largest validation error within the entire validation data. 
    '''
    
    # set threshold as global max of validation errors
    threshold = train_errors.max().item()

    # set test scores as max error in one time tick
    score, _ = test_errors.max(dim=0)

    return score > threshold

def _mean_thresholding(train_errors, test_errors):
    '''
    Returns anomaly predictions on test errors based on threshold
    calculated on the validation errors. 
    Threshold is the largest validation error within the entire validation data. 
    '''
    
    # set threshold as global max of validation errors
    threshold = train_errors.mean(dim=0).max().item()

    # set test scores as max error in one time tick
    score = test_errors.mean(dim=0)

    return score > threshold

def _best_thresholding(test_errors, test_labels):
    '''
    Returns anomaly predictions on test errors based on threshold
    calculated on the validation errors. 
    Threshold is the largest validation error within the entire validation data. 

    ONLY USE TO TEST THEORETICAL PERFORMANCE, NOT FOR REAL EVALUATION!
    '''
    
    # set threshold as global max of validation errors   

    max_score, _ = test_errors.max(dim=0)
    mean_score = test_errors.mean(dim=0)
    scores = {'max': max_score, 'mean': mean_score}

    best_f1 = 0
    best_method = None
    best_predictions = None

    lower_bound = min(min(max_score), min(mean_score)).item()
    upper_bound = max(max(max_score), max(mean_score)).item()

    thresholds = np.linspace(lower_bound, upper_bound, 1000)
    for threshold in thresholds:
        for method, score in scores.items():
            anomaly_predictions = score > threshold
            precision, recall = precision_recall(anomaly_predictions, test_labels)
            f1 = F_score(precision, recall, beta=1)
            if f1 > best_f1:
                best_f1 = f1
                best_method = method
                best_predictions = anomaly_predictions
    
    return best_predictions, best_method