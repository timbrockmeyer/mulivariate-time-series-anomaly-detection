### TODO:NEEDS REWORK !!!


import numpy as np
import torch
import seaborn as sns
import pandas as pd
from ..utils.utils import normalize_with_median_iqr

def get_error_distribution_plot(results_dict):
    '''
    Returns a plot of the error distribution derived from predictions 
    and groundtruth values.

    Args:
        results_dict (dict): Dictionary of test results.
    '''
    
    errors = []
    for key, value in results_dict.items():
        y_pred, y, _ = value
        err = torch.abs(y_pred - y).cpu().numpy()
        err = normalize_with_median_iqr(err)
        s = pd.Series(err, index=[key]*len(err))
        errors.append(s)
        if key == 'Validation':
            threshold = err.max()

    errors = pd.Series(dtype=np.float64).append(errors)
    errors = errors.apply(lambda x: np.nan if x < threshold*0.75 else x)
    df = pd.DataFrame({'normalized_error': errors})
    df = df.dropna()
    df.index.name = 'Mode'
    df = df.reset_index()

    error_plot = sns.displot(df, x="normalized_error", hue="Mode", kind="kde", fill=True)
    return error_plot
    
