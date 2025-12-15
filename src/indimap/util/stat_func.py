import numpy as np
from scipy.optimize import curve_fit


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


def exponential_func(x, a, b):
    """Exponential function for curve fitting."""
    return a * np.exp(b * x)


def fit_expo(data):
    x = np.arange(len(data))
    y = np.array(data)

    try: 
        popt, _ = curve_fit(exponential_func, x, y, p0=(1, 0.01), maxfev=10000)
        a, b = popt
        return a, b
    except RuntimeError:
        return None, None