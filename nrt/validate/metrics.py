"""Various metrics to assess performances of monitoring algorithms
"""
import datetime
import functools
import numpy as np
from kneed import KneeLocator


def _sigmoid_f1_curve(y_true, y_pred, begin, tolerance_window=(-30, 365),
                      sample_weight=None, step=5):
    """F1 curve at increasing time lag

    Args:
        y_true (array-like): Reference disturbance dates (in days since epoch)
            or zeros for no disturbance.
        y_pred (array-like): Predicted disturbance dates (in days since epoch)
            or zeros for no detection.
        begin (int or datetime): Beginning date to compute lag for FP.
        tolerance_window (tuple): Time window to assess true positives.
        sample_weight (array-like or None): Weights for samples, if provided.
        step (int): Increment for the lag.

    Examples:
        >>> import datetime
        >>> import random
        >>> import matplotlib.pyplot as plt
        >>> y_true = [0] * 100 + [19510 + random.randint(0, 350) for _ in range(100)]
        >>> y_pred = [0] * 80 + [19510 + random.randint(0, 350) for _ in range(20)] + \
                     [y_true[100 + i] + random.randint(-40, 400) for i in range(80)] + [0] * 20
        >>> lags, f1s = _sigmoid_f1_curve(y_true=y_true, y_pred=y_pred,
        ...                   begin=datetime.datetime(2023, 6, 1))
        >>> plt.plot(lags, f1s)
        >>> plt.show()

    Returns:
        tuple: A tuple of two arrays: lags and corresponding F1 scores.

    """
    if isinstance(begin, datetime.datetime):
        begin = (begin - datetime.datetime(1970, 1, 1)).days
    y_true = np.array(y_true).astype(np.float32)
    y_pred = np.array(y_pred).astype(np.float32)

    # Prepare array of potential TP, TN, FN with lag values
    TPs_mask = np.logical_and(y_true != 0, y_pred != 0)
    TPs_ = (y_pred - y_true)[TPs_mask]

    FPs_mask = y_pred != 0
    y_true_modified = np.where(y_true == 0, begin, y_true)
    FPs_ = (y_pred - y_true_modified)[FPs_mask]

    FNs_mask = y_true != 0
    FNs_ = np.where(y_pred == 0, np.inf, y_pred - y_true)
    FNs_ = FNs_[FNs_mask]

    f1s = []
    lags = []
    for lag in range(*tolerance_window, step):
        TP = (TPs_ < lag).sum()
        FP = (FPs_ > lag).sum()
        FN = (FNs_ > lag).sum()
        if 2 * TP + FP + FN == 0:
            f1 = 0.0
        else:
            f1 = 2 * TP / (2 * TP + FP + FN)
        f1s.append(f1)
        lags.append(lag)

    return lags, f1s


def sigmoid_initial_delay(y_true, y_pred, begin, tolerance_window=(-30, 365),
                          sample_weight=None, step=5):
    """Compute initial delay lag and F1 score

    Note:
        This implementation is based on the F1 curve and may diverge from the method
        described in Bullock et al. 2022, which uses omission errors to compute the initial delay.
        This is a simplification that uses the F1 score to approximate the initial delay.

    Args:
        y_true (array-like): Reference disturbance dates (in days since epoch)
            or zeros for no disturbance.
        y_pred (array-like): Predicted disturbance dates (in days since epoch)
            or zeros for no detection.
        begin (int or datetime): Beginning date to compute lag for FP.
        tolerance_window (tuple): Lag window within which detection is considered a TP.
        sample_weight (array-like or None): Sample weights.
        step (int): Step increment for the lag.

    Returns:
        tuple: (lag, f1) at the initial delay where F1 exceeds 0.05

    Examples:
        >>> import datetime
        >>> import random
        >>> y_true = [0] * 100 + [19510 + random.randint(0, 350) for _ in range(100)]
        >>> y_pred = [0] * 80 + [19510 + random.randint(0, 350) for _ in range(20)] + \
                     [y_true[100 + i] + random.randint(-40, 400) for i in range(80)] + [0] * 20
        >>> sigmoid_initial_delay(y_true=y_true, y_pred=y_pred,
        ...                       begin=datetime.datetime(2023, 6, 1))
    """
    lags, f1s = _sigmoid_f1_curve(y_true=y_true, y_pred=y_pred, begin=begin,
                                  tolerance_window=tolerance_window,
                                  sample_weight=sample_weight, step=step)

    # Find initial delay where F1 exceeds 0.05
    f1s = np.array(f1s)
    lags = np.array(lags)
    # Get the indices where F1 is greater than 0.05
    valid_indices = np.where(f1s > 0.05)[0]
    if valid_indices.size > 0:
        initial_delay_index = valid_indices[0]
        return int(lags[initial_delay_index]), float(f1s[initial_delay_index])
    return 0, 0.0


def sigmoid_level_off(y_true, y_pred, begin, tolerance_window=(-30, 365),
                      sample_weight=None, step=5):
    """Compute the level-off point for F1 curve using a knee-detection method.

    Args:
        y_true (array-like): Reference disturbance dates (in days since epoch)
            or zeros for no disturbance.
        y_pred (array-like): Predicted disturbance dates (in days since epoch)
            or zeros for no detection.
        begin (int or datetime): Beginning date to compute lag for FP.
        tolerance_window (tuple): Lag window within which detection is considered a TP.
        sample_weight (array-like or None): Sample weights.
        step (int): Step increment for the lag.

    Examples:
        >>> import datetime
        >>> import random
        >>> y_true = [0] * 100 + [19510 + random.randint(0, 350) for _ in range(100)]
        >>> y_pred = [0] * 80 + [19510 + random.randint(0, 350) for _ in range(20)] + \
                     [y_true[100 + i] + random.randint(-40, 400) for i in range(80)] + [0] * 20
        >>> sigmoid_level_off(y_true=y_true, y_pred=y_pred,
        ...                   begin=datetime.datetime(2023, 6, 1))

    Returns:
        tuple: (lag, f1) at the level-off point. If the level-off point exceeds the maximum 
               tolerance window, it is capped at tolerance_window[1].

    """
    lags, f1s = _sigmoid_f1_curve(y_true=y_true, y_pred=y_pred, begin=begin,
                                  tolerance_window=tolerance_window,
                                  sample_weight=sample_weight, step=step)

    # Use the KneeLocator to find the "knee" or "level-off" point in the F1 curve
    knee_locator = KneeLocator(lags, f1s, curve="concave", direction="increasing")
    level_off_lag = knee_locator.knee

    # If level-off point is greater than the upper bound of tolerance_window, cap it
    if level_off_lag is not None and level_off_lag > tolerance_window[1]:
        level_off_lag = tolerance_window[1]

    if level_off_lag is not None:
        index = np.where(lags == level_off_lag)[0][0]
        return lags[index], float(f1s[index])

    return tolerance_window[1], float(f1s[-1])


def f1_score_at_lag(y_true, y_pred, lag, begin, negative_tolerance=50,
                    sample_weight=None):
    """F1 score at a given lag threshold.

    Args:
        y_true (array-like): Reference disturbance dates (in days since epoch)
            or zeros for no disturbance.
        y_pred (array-like): Predicted disturbance dates (in days since epoch)
            or zeros for no detection.
        lag (int): The lag at which to compute the F1 score.
        begin (int or datetime): Beginning date to compute lag for FP.
        negative_tolerance (int): Allowed negative tolerance for predictions before events.
        sample_weight (array-like or None): Sample weights.

    Examples:
        >>> import datetime
        >>> y_true = [0, 19510, 0, 0, 19512, 0, 19520, 19525, 0, 19530]
        >>> y_pred = [0, 19512, 0, 0, 0, 0, 19523, 19528, 0, 19535]
        >>> f1_score_at_lag(y_true=y_true, y_pred=y_pred, lag=10,
        ...                 begin=datetime.datetime(2023, 1, 1))
        0.8

    Returns:
        f1 (float): The F1 score at the specified lag.

    """
    if isinstance(begin, datetime.datetime):
        begin = (begin - datetime.datetime(1970, 1, 1)).days

    y_true = np.array(y_true).astype(np.float32)
    y_pred = np.array(y_pred).astype(np.float32)

    # Prepare masks for TP, FP, and FN
    TPs_mask = np.logical_and(y_true != 0, y_pred != 0)
    TPs_ = (y_pred - y_true)[TPs_mask]

    FPs_mask = y_pred != 0
    y_true_modified = np.where(y_true == 0, begin, y_true)
    FPs_ = (y_pred - y_true_modified)[FPs_mask]

    FNs_mask = y_true != 0
    FNs_ = np.where(y_pred == 0, np.inf, y_pred - y_true)
    FNs_ = FNs_[FNs_mask]

    TP = (TPs_ < lag).sum()
    FP = (FPs_ > lag).sum()
    FN = (FNs_ > lag).sum()

    if 2 * TP + FP + FN == 0:
        return 0.0
    else:
        return (2 * TP / (2 * TP + FP + FN)).item()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

