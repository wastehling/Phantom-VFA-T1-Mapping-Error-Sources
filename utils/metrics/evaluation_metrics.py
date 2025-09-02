import numpy as np


def normalized_RMSE(y_true, y_pred, range):
    """
    Normalized RMSE between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.mean(((y_true - y_pred) / range) ** 2))


def normalized_R99SE(y_true, y_pred, range):
    """
    Normalized root of the 5th percentile of the squared error between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.percentile((((y_true - y_pred) / range) ** 2), 99))


def normalized_R95SE(y_true, y_pred, range):
    """
    Normalized root  of the 95th percentile of the squared error between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.percentile((((y_true - y_pred) / range) ** 2), 95))

def normalized_R90SE(y_true, y_pred, range):
    """
    Normalized root  of the 95th percentile of the squared error between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.percentile((((y_true - y_pred) / range) ** 2), 90))


def normalized_R75SE(y_true, y_pred, range):
    """
    Normalized root  of the 95th percentile of the squared error between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.percentile((((y_true - y_pred) / range) ** 2), 75))


def normalized_R50SE(y_true, y_pred, range):
    """
    Normalized root  of the 95th percentile of the squared error between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.percentile((((y_true - y_pred) / range) ** 2), 75))


def normalized_R25SE(y_true, y_pred, range):
    """
    Normalized root  of the 95th percentile of the squared error between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.percentile((((y_true - y_pred) / range) ** 2), 25))


def normalized_R10SE(y_true, y_pred, range):
    """
    Normalized root  of the 95th percentile of the squared error between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.percentile((((y_true - y_pred) / range) ** 2), 10))

def normalized_R5SE(y_true, y_pred, range):
    """
    Normalized root of the 5th percentile of the squared error between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.percentile((((y_true - y_pred) / range) ** 2), 5))


def normalized_R1SE(y_true, y_pred, range):
    """
    Normalized root of the 5th percentile of the squared error between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized RMSE
    """
    return np.sqrt(np.percentile((((y_true - y_pred) / range) ** 2), 1))


def normalized_MAE(y_true, y_pred, range):
    """
    Normalized MAE between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        normalized MAE
    """
    return np.mean(np.abs((y_true - y_pred) / range))


def SD_error(y_true, y_pred, range):
    """
    Standard deviation between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        standard deviation
    """
    return np.std((y_true - y_pred))


def bias(y_true, y_pred, range):
    """
    Bias between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        bias
    """
    return np.mean((y_true - y_pred))


def variance(y_true, y_pred, range):
    """
    Variance between y_true and y_pred
    Args:
        y_true: ground truth
        y_pred: prediction
        range: normalization range

    Returns:
        variance
    """
    return np.var((y_true - y_pred))



