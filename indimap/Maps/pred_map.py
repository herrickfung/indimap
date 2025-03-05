'''
contains all functions related to prediction analyses

TDL:
3. plot graph

MAYDO: 
1. consider standardizing the data before fitting models
'''

from itertools import combinations, permutations
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from pathlib import Path
import numpy as np

from .corr_map import CorrMap
from .util import map_func, pred_func

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
        met_type = ['within', 'across']
        sources = ['subj', 'inst']
        methods = ['rand', 'avg', 'corr', 'ols', 'lasso', 'ridge']
        self.pred_results = {
            met: {
            source: {
                method: None for method in methods
            } for source in sources
            } for met in met_type
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

        met_type = ['within', 'across']
        source_arr = ['subj', 'inst']
        for met in met_type:
            for source in source_arr:
                self.pred_from_rand(met, source)
                self.pred_from_avg(met, source)
                self.pred_from_corr(met, source)
                # self.pred_from_fit(met, source)

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

        self.split_raw_mat = {  # for rand, avg, corr
            'subj': human_split,
            'inst': model_split
        }
        self.raw_mat = {   # for ensemble fitting
            'subj': human_arr,
            'inst': model_arr
        }

    def pred_from_rand(self, within: str, source: str) -> None:
        """ 
        Perform prediction analyses from random data 
        Take a random subject/instance from the same split to predict
        either within or across metric of the target subject
        """

        # select only the second split 
        X = self.split_raw_mat[source][:, 1, :, :, :]
        Y = self.split_raw_mat['subj'][:, 1, :, :, :]

        # handle within and across metric prediction
        n_bs, n_met, n_subjs, _ = Y.shape
        if within == 'within':
            met_pairs = [(i, i) for i in range(n_met)]  # this allow shared logic
            output = np.empty((n_bs, n_met, n_subjs))
        else:
            met_pairs = list(permutations(range(n_met), 2))
            n_met_pairs = len(met_pairs)
            output = np.empty((n_bs, n_met_pairs, n_subjs))

        # prediction and evaluation loop
        np.random.seed(self.bs_seed)
        for bs in range(n_bs):
            for met, (met_a, met_b) in enumerate(met_pairs):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    rand_subj = np.random.choice(other_subjs)
                    x = X[bs, met_a, rand_subj, :]
                    y = Y[bs, met_b, subj, :]
                    output[bs, met, subj] = np.corrcoef(x, y)[0, 1]

        self.pred_results[within][source]['rand'] = output

    def pred_from_avg(self, within : str, source : str) -> None:
        """ Perform prediction analyses from average data 
        Take the average of all other subjects/instances to predict
        """

        # select only the second split
        X = self.split_raw_mat[source][:, 1, :, :, :]
        Y = self.split_raw_mat['subj'][:, 1, :, :, :]

        # handle within and across metric prediction
        n_bs, n_met, n_subjs, _ = Y.shape
        if within == 'within':
            met_pairs = [(i, i) for i in range(n_met)] 
            output = np.empty((n_bs, n_met, n_subjs))
        else:
            met_pairs = list(permutations(range(n_met), 2))
            n_met_pairs = len(met_pairs)
            output = np.empty((n_bs, n_met_pairs, n_subjs))

        for bs in range(n_bs):
            for met, (met_a, met_b) in enumerate(met_pairs):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    x = np.nanmean(X[bs, met_a, other_subjs, :], axis=0)
                    y = Y[bs, met_b, subj, :]
                    output[bs, met, subj] = np.corrcoef(x, y)[0, 1]

        self.pred_results[within][source]['avg'] = output

    def pred_from_corr(self, within : str, source : str) -> None:
        """ Perform prediction analyses from CorrMap 
        Use the correlation map as weight to predict the target subject/instance
        """

        # select only the second split for data
        X = self.split_raw_mat[source][:, 1, :, :, :]
        Y = self.split_raw_mat['subj'][:, 1, :, :, :]
        # get weight from corr map first split
        W = self.corr_map[source][:, 0, :, :, :]

        # handle within and across metric prediction
        n_bs, n_met, n_subjs, _ = Y.shape
        if within == 'within':
            met_pairs = [(i, i) for i in range(n_met)] 
            output = np.empty((n_bs, n_met, n_subjs))
        else:
            met_pairs = list(permutations(range(n_met), 2))
            n_met_pairs = len(met_pairs)
            output = np.empty((n_bs, n_met_pairs, n_subjs))

        for bs in range(n_bs):
            for met, (met_a, met_b) in enumerate(met_pairs):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    x = X[bs, met_a, other_subjs, :]
                    w = W[bs, met_a, subj, :] if source == 'subj' \
                        else W[bs, met_a, subj, other_subjs] # arbitarily remove 1 inst to match shape
                    y = Y[bs, met_b, subj, :]
                    y_pred =  w @ x
                    output[bs, met, subj] = np.corrcoef(y_pred, y)[0, 1]

        self.pred_results[within][source]['corr'] = output

    def pred_from_fit(self, within : str, source : str) -> None:
        """ Perform prediction analyses from fitting """

        methods = ['ols', 'lasso', 'ridge']

        for method in methods:
            # init, train on first split
            if method == 'ols':
                model = LinearRegression()
            elif method == 'lasso':
                model = Lasso()
            elif method == 'ridge':
                model = Ridge()

            X = self.raw_mat[source][:, 0, :, :, :]
            W, C, A = pred_func.train_model_for_each(X, model=model)

            # test on second split and get prediction
            X = self.raw_mat[source][:, 1, :, :, :]
            Y = self.raw_mat['subj'][:, 1, :, :, :]
            pred_acc = pred_func.test_model_for_each(X, Y, W, C, A, model=model)

            self.pred_results[within][source][method] = pred_acc