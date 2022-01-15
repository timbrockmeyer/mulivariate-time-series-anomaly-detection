import torch
from . import utils
    
def precision_recall(pred, labels):
    ''' 
    Calculates precision and recall.
    Precision = (TP / (TP+FP)).
    Recall = (TP / (TP+FN)).

    Args:
        pred (Tensor): 1-dimensional tensor of predictions.
        labels (Tensor): 1-dimensional tensor of ground truth observations.
    '''
    pred, labels = utils.cast(torch.bool, pred, labels)
    
    # precision
    hits = labels[pred]
    precision = hits.sum() / pred.sum()
    
    # recall
    hits = pred[labels]
    recall = hits.sum() / labels.sum()
    
    return precision.item(), recall.item()

def F_score(precision, recall, beta=1):
    ''' 
    Calculates F-scores.
    
    Args:
        precision (int, float): Precision score.
        recall (int, float): Recall score.
        beta (int, float, optional): Positive number.
    '''
    div = (beta**2 * precision) + recall
    if div > 0:
        return ((1 + beta**2) * (precision * recall)) / div
    else:
        return 0
        