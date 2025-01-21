import numpy as np
from sklearn.metrics import matthews_corrcoef

class stat_func:
    @staticmethod
    def regress(x, y, metric):
        if metric == 'pearson':
            result = np.corrcoef(x, y)[0, 1]
        return result


    @staticmethod
    def r2z(r, metric):
        '''convert correlation to z-score'''
        if metric == 'pearson':
            result = 0.5 * (np.log(1 + r) - np.log(1 - r))
            if np.isinf(result).any():
                result[np.isinf(result)] = 1
        else:
            result = r
        return result


    @staticmethod
    def z2r(z, metric):
        '''convert z-score to correlation'''
        if metric == 'pearson':
            result = (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
        else:
            result = z
        return result


