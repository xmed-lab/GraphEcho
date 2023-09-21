import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score
)

from monai.data import (
    decollate_batch,
)

from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def C_index(y_true, y_pred):
    return concordance_index(y_true, y_pred) * 100

def DiceScore(y_true, y_pred):
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    labels_list = decollate_batch(y_true)
    labels_convert = [post_label(label) for label in labels_list]
    outputs_list = decollate_batch(y_pred)
    outputs_convert = [post_pred(output) for output in outputs_list]
    dice_metric(y_pred=outputs_convert, y=labels_convert)
    dice = dice_metric.aggregate().item()
    return dice
