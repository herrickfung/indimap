'''
contains all functions related to prediction analyses

TDL:
1. make it capatible with Lasso and Ridge
2. edit for cross metric predictions
3. plot graph
'''

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from pathlib import Path
from scipy.stats import sem
import numpy as np

from .corr_map import CorrMap
from .util import map_func

class PredMap:
    def __init__(self, config):
        self.config = config
        self.human = self.config.get('subj_data')
        self.model = self.config.get('inst_data')
        self.human_iden = self.config.get('subj_column_name')
        self.model_iden = self.config.get('inst_column_name')
        self.map_var = self.config.get('map_variables')
        self.map_tgt = self.config.get('map_together')
        self.map_sep = self.config.get('map_separate')
        self.n_bs = self.config.get('bootstrap_iterations')
        self.bs_seed = self.config.get('bootstrap_seed')
        self.output_path = Path(self.config['output_path'])
        self.graph_path = Path(self.config['graph_path'])

        # initialize output
        sources = ['subj', 'inst']
        methods = ['rand', 'avg', 'corr', 'fit']
        self.pred_results = {source: {method: None for method in methods} 
                             for source in sources
                             }

    def check_exist(self):
        """ check whether the file exist"""
        file_path = self.output_path / 'PredMap_results.npz'
        return file_path.exists()

    def load_all(self, path):
        """ load all the results from the file"""
        self.pred_results = np.load(path / 'PredMap_results.npz', 
                                    allow_pickle=True
                                    )

    def save_all(self, path):
        """ save all the results to the file"""
        output_path = path / 'PredMap_results.npz'
        np.savez(output_path, **self.pred_results)

    def compute_pred_maps(self):
        """ Perform prediction analyses """

        # setup
        self.compute_corr_map()
        self.compute_raw_mat()

        source_arr = ['subj', 'inst']
        for source in source_arr:
            # self.pred_from_rand(source)
            # self.pred_from_avg(source)
            # self.pred_from_corr(source)
            self.pred_from_fit(source)
            exit()
        print('done')
        
    def compute_corr_map(self):
        """ Load CorrMap object """
        self.corr_map = CorrMap(self.config)
        try:
            self.corr_map.load_map()
        except FileNotFoundError:
            self.corr_map = self.corr_map.compute_corr_maps()

        self.corr_map = {
            'subj': self.corr_map.corr_maps['subj_to_subj'],
            'inst': self.corr_map.corr_maps['subj_to_inst']
        }

    def compute_raw_mat(self):
        """ Get split raw data matrix """
        human_arr = map_func.convert_to_array(self.human, self.human_iden,
                                              self.map_var, self.map_tgt, 
                                              self.map_sep
                                              )
        model_arr = map_func.convert_to_array(self.model, self.model_iden,
                                              self.map_var, self.map_tgt,
                                              self.map_sep
                                              )
        map_func.check_for_extreme(human_arr, model_arr)

        human_split, model_split = map_func.split_arr(human_arr,
                                                      model_arr,
                                                      self.n_bs,
                                                      self.bs_seed
                                                      )

        human_split = np.nanmean(human_split, axis = 2)  # average across conditions.
        model_split = np.nanmean(model_split, axis = 2)  # average across conditions.

        self.raw_mat = {
            'subj': human_split,
            'inst': model_split
        }

    def pred_from_rand(self, source='subj') -> None:
        """ Perform prediction analyses from random data """

        # select only the second split
        X = self.raw_mat[source][:, 1, :, :, :]
        Y = self.raw_mat['subj'][:, 1, :, :, :]

        n_bs, n_met, n_subjs, _ = Y.shape
        output = np.empty((n_bs, n_met, n_subjs))

        np.random.seed(self.bs_seed)
        for bs in range(n_bs):
            for met in range(n_met):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    rand_subj = np.random.choice(other_subjs)
                    x = X[bs, met, rand_subj, :]
                    y = Y[bs, met, subj, :]
                    output[bs, met, subj] = np.corrcoef(x, y)[0, 1]

        self.pred_results[source]['rand'] = output

    def pred_from_avg(self, source='subj') -> None:
        """ Perform prediction analyses from average data """

        # select only the second split
        X = self.raw_mat[source][:, 1, :, :, :]
        Y = self.raw_mat['subj'][:, 1, :, :, :]

        n_bs, n_met, n_subjs, _ = Y.shape
        output = np.empty((n_bs, n_met, n_subjs))

        for bs in range(n_bs):
            for met in range(n_met):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    x = np.nanmean(X[bs, met, other_subjs, :], axis=0)
                    y = Y[bs, met, subj, :]
                    output[bs, met, subj] = np.corrcoef(x, y)[0, 1]

        self.pred_results[source]['avg'] = output

    def pred_from_corr(self, source='subj') -> None:
        """ Perform prediction analyses from CorrMap """

        # select only the second split for data
        X = self.raw_mat[source][:, 1, :, :, :]
        Y = self.raw_mat['subj'][:, 1, :, :, :]
        # get weight from corr map first split
        W = self.corr_map[source][:, 0, :, :, :]

        n_bs, n_met, n_subjs, _ = Y.shape
        output = np.empty((n_bs, n_met, n_subjs))

        for bs in range(n_bs):
            for met in range(n_met):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    x = X[bs, met, other_subjs, :]
                    w = W[bs, met, subj, :] if source == 'subj' \
                        else W[bs, met, subj, other_subjs] # arbitarily remove 1 inst to match shape
                    y = Y[bs, met, subj, :]
                    y_pred =  w @ x
                    output[bs, met, subj] = np.corrcoef(y_pred, y)[0, 1]

        self.pred_results[source]['corr'] = output

    def pred_from_fit(self, source='subj') -> None:
        """ Perform prediction analyses from fitting """

        # train weights on first split
        X = self.raw_mat[source][:, 0, :, :, :]
        n_bs, n_met, n_subjs, _ = X.shape
        W = np.empty((n_bs, n_met, n_subjs, n_subjs - 1))
        C = np.empty((n_bs, n_met, n_subjs))
        for bs in range(n_bs):
            for met in range(n_met):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    C[bs, met, subj], W[bs, met, subj] = \
                        self.train_weights(
                            X[bs, met, other_subjs, :], 
                            X[bs, met, subj, :],
                            model=LinearRegression(),
                        )

        # predict on second split
        X = self.raw_mat[source][:, 1, :, :, :]
        Y = self.raw_mat['subj'][:, 1, :, :, :]

        n_bs, n_met, n_subjs, _ = Y.shape
        output = np.empty((n_bs, n_met, n_subjs))

        for bs in range(n_bs):
            for met in range(n_met):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    x = X[bs, met, other_subjs, :]
                    w = W[bs, met, subj, :]
                    c = C[bs, met, subj]
                    y = Y[bs, met, subj, :]
                    y_pred = self.eval_weights(x, w, c,
                                               model=LinearRegression(),
                                               )
                    output[bs, met, subj] = np.corrcoef(y_pred, y)[0, 1]

        self.pred_results[source]['fit'] = output

    @staticmethod
    def train_weights(X, Y, model=LinearRegression()) -> np.ndarray:
        """ Do a five-fold cross validation to train weights for each subj """
        n_subjs, n_imgs = X.shape
        stims = np.arange(n_imgs)

        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        Intercept = np.empty((k, n_subjs))
        Beta = np.empty((k, n_subjs))

        for fold, (train_index, _) in enumerate(kf.split(stims)):
            X_train = X[:, train_index].T
            Y_train = Y[train_index]
            model.fit(X_train, Y_train)
            Intercept[fold] = model.intercept_
            Beta[fold] = model.coef_

        Intercept = np.mean(Intercept)
        Beta = np.mean(Beta, axis=0)
        return Intercept, Beta

    @staticmethod
    def eval_weights(x, w, c, model=LinearRegression()) -> np.ndarray:
        """ Do evaluation of the predictivity using the fitted weights """
        model.coef_ = w
        model.intercept_ = c
        y_pred = model.predict(x.T)
        return y_pred





