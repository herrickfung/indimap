import numpy as np


def r2z(r: np.ndarray, metric: str) -> np.ndarray:
    '''convert correlation to z-score'''
    if metric == 'pearson':
        result = 0.5 * (np.log(1 + r) - np.log(1 - r))
        if np.isnan(result).any():
            result[np.isnan(result)] = 10
    else:
        result = r
    return result


def z2r(z: np.ndarray, metric: str) -> np.ndarray:
    '''convert z-score to correlation'''
    if metric == 'pearson':
        result = (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
    else:
        result = z
    return result


