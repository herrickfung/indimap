'''
contains all functions related to prediction analyses
'''

from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib import rcParams
from itertools import permutations
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy.stats import sem
from pathlib import Path
import numpy as np
rcParams['font.family'] = 'CMU Sans Serif'

plt.set_loglevel('error')

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from .corr_map import CorrMap
from indimap.util import map_func, pred_func


class PredMap:
    def __init__(self, config: dict):
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
        methods = ['rand', 'avg', 'corr']
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
        loaded= np.load(path / 'PredMap_results.npz', 
                        allow_pickle=True
                        )
        self.pred_results['within'] = loaded['within'].item()
        self.pred_results['across'] = loaded['across'].item()

    def save_all(self, path):
        """ save all the results to the file"""
        output_path = path / 'PredMap_results.npz'
        np.savez(output_path, **self.pred_results)

    def compute_pred_maps(self):
        """ Perform prediction analyses """

        # # setup
        self.compute_corr_map()
        self.compute_raw_mat()

        # loop to run all analyses
        met_type = ['within', 'across']
        source_arr = ['subj', 'inst']
        for met in met_type:
            for source in source_arr:
                self.pred_from_rand(met, source)
                self.pred_from_avg(met, source)
                self.pred_from_corr(met, source)

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

        image_splits = np.empty(shape=(self.n_bs, 2, int(human_arr.shape[-2]/2)))
        for i, (split1, split2) in enumerate(map_func.split_image_array(
            human_arr,
            model_arr,
            self.n_bs,
            self.bs_seed
        )):
            image_splits[i, 0] = split1
            image_splits[i, 1] = split2

        # average across conditions
        human_arr = human_arr.mean(axis=0)
        model_arr = model_arr.mean(axis=0)

        self.image_split = {  # for rand, avg, corr
            'train': image_splits[:, 0, :],
            'test': image_splits[:, 1, :]
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

        X = self.raw_mat[source]
        Y = self.raw_mat['subj']
        n_met, n_subjs, _, _ = Y.shape

        # handle within and across metric prediction

        if within == 'within':
            met_pairs = [(i, i) for i in range(n_met)]  # this allow shared logic
            output = np.empty((self.n_bs, n_met, n_subjs))
        else:
            met_pairs = list(permutations(range(n_met), 2))
            n_met_pairs = len(met_pairs)
            output = np.empty((self.n_bs, n_met_pairs, n_subjs))

        # prediction and evaluation loop
        np.random.seed(self.bs_seed)
        for bs in range(self.n_bs):
            bs_img_split = self.image_split['test'][bs].astype(int)
            for met, (met_a, met_b) in enumerate(met_pairs):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    rand_subj = np.random.choice(other_subjs)

                    x = X[met_a, rand_subj, bs_img_split, 0]
                    y = Y[met_b, subj, bs_img_split, 0]
                    output[bs, met, subj] = np.corrcoef(x, y)[0, 1]

        self.pred_results[within][source]['rand'] = output

    def pred_from_avg(self, within : str, source : str) -> None:
        """ Perform prediction analyses from average data 
        Take the average of all other subjects/instances to predict
        """

        # handle within and across metric prediction
        n_met, n_subjs, _, _ = self.raw_mat['subj'].shape
        if within == 'within':
            met_pairs = [(i, i) for i in range(n_met)] 
            output = np.empty((self.n_bs, n_met, n_subjs))
        else:
            met_pairs = list(permutations(range(n_met), 2))
            n_met_pairs = len(met_pairs)
            output = np.empty((self.n_bs, n_met_pairs, n_subjs))

        for bs in range(self.n_bs):
            bs_img_split = self.image_split['test'][bs].astype(int)
            X = self.raw_mat[source][:, :, bs_img_split, 0]
            Y = self.raw_mat['subj'][:, :, bs_img_split, 0]
            for met, (met_a, met_b) in enumerate(met_pairs):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    x = np.nanmean(X[met_a, other_subjs], axis=0)
                    y = Y[met_b, subj]
                    output[bs, met, subj] = np.corrcoef(x, y)[0, 1]

        self.pred_results[within][source]['avg'] = output

    def pred_from_corr(self, within : str, source : str) -> None:
        """ Perform prediction analyses from CorrMap 
        Use the correlation map as weight to predict the target subject/instance
        """

        # get weight from corr map first split
        W = self.corr_map[source][:, 0, :, :, :]

        # handle within and across metric prediction
        n_met, n_subjs, _, _ = self.raw_mat['subj'].shape
        if within == 'within':
            met_pairs = [(i, i) for i in range(n_met)] 
            output = np.empty((self.n_bs, n_met, n_subjs))
        else:
            met_pairs = list(permutations(range(n_met), 2))
            n_met_pairs = len(met_pairs)
            output = np.empty((self.n_bs, n_met_pairs, n_subjs))

        for bs in range(self.n_bs):
            bs_img_split = self.image_split['test'][bs].astype(int)
            X = self.raw_mat[source][:, :, bs_img_split, 0]
            Y = self.raw_mat['subj'][:, :, bs_img_split, 0]
            for met, (met_a, met_b) in enumerate(met_pairs):
                for subj in range(n_subjs):
                    other_subjs = [i for i in range(n_subjs) if i != subj]
                    x = X[met_a, other_subjs]
                    w = W[bs, met_a, subj, :] if source == 'subj' \
                        else W[bs, met_a, subj, other_subjs] # arbitarily remove 1 inst to match shape
                    y = Y[met_b, subj]
                    y_pred =  w @ x
                    output[bs, met, subj] = np.corrcoef(y_pred, y)[0, 1]

        self.pred_results[within][source]['corr'] = output

    def pred_from_fit(self, within : str, source : str) -> None:
        """ Perform prediction analyses from fitting """

        methods = ['ols', 'lasso', 'ridge']
        for method in methods:
            if method == 'ols':
                model = LinearRegression()
            elif method == 'lasso':
                model = Lasso(max_iter=1000)
            elif method == 'ridge':
                model = Ridge(max_iter=1000)

            X = self.raw_mat[source]
            Y = self.raw_mat['subj']

            n_conds, n_met, n_subjs, n_imgs = X.shape
            if within == 'within':
                met_pairs = [(i, i) for i in range(n_met)]
            else:
                met_pairs = list(permutations(range(n_met), 2))
            n_met_pairs = len(met_pairs)

            # init k fold and results
            k = 5
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            stims = np.arange(n_imgs)
            pred_acc_arr = np.empty((k, n_conds, n_met_pairs, n_subjs))

            for fold, (train_idx, test_idx) in enumerate(kf.split(stims)):
                # training
                x_train = X[:, :, :, train_idx]; y_train = Y[:, :, :, train_idx]
                W, C, A = pred_func.train_model_for_each(
                    model=model, 
                    X_train=x_train, 
                    Y_train=y_train,
                    within = within,
                )

                # testing
                x_test = X[:, :, :, test_idx]; y_test = Y[:, :, :, test_idx]
                pred_acc = pred_func.test_model_for_each(
                    model=model,
                    X_test=x_test,
                    Y_test=y_test,
                    W=W,
                    C=C,
                    A=A,
                    within = within,
                )
                pred_acc_arr[fold] = pred_acc

            # average across folds and write results
            output = np.nanmean(pred_acc_arr, axis=0)  # n_conds, n_met_pars, n_subjs
            self.pred_results[within][source][method] = output

    def plot_wn_var(self):
        """ plot results for within metric prediction """

        # init
        data = self.pred_results['within']
        sources = ['subj', 'inst']
        methods = ['rand', 'avg', 'corr']
        n_sources = len(sources)
        n_methods = len(methods)
        n_bs, n_met, n_subjs = data['subj']['rand'].shape

        # setup plot data
        trans_data = np.empty((n_sources, n_methods, n_met, n_subjs))
        for i, sor in enumerate(sources):
            for j, met in enumerate(methods):
                sub_data = data[sor][met]
                # average across bs for first 3 method, average across condition for other
                sub_data = np.nanmean(sub_data, axis=0)  
                trans_data[i, j] = sub_data
        trans_data = rearrange(trans_data, 's m met subj -> met s m subj')

        # plot
        plt.clf()
        fig, axs = plt.subplots(1, n_met, figsize=(n_met * 10/3, 4))
        colors = plt.cm.get_cmap('Dark2', 8)
        labels = ['Predict from Subject', 'Predict from Instance']
        method_labels = ['Rand', 'Avg', 'Corr']

        for i, met in enumerate(self.map_var):
            ax = axs[i]
            for j, sor in enumerate(sources):
                for k, method in enumerate(methods):
                    x_pos = j * 0.8 + k * 3
                    ax.bar(x_pos,
                           np.mean(trans_data[i,j,k,:]),
                           yerr = sem(trans_data[i,j,k,:]),
                           color = colors(j),
                           alpha = 0.5,
                           label = labels[j] if k == 0 else None,
                           )
                    ax.scatter([x_pos - 0.25 for _ in range(n_subjs)],
                               trans_data[i,j,k,:],
                               color = colors(j),
                               s = 5,
                            )

            ax.set_ylim(-0.1, 1)
            ax.set_title(met, fontsize=14)
            ax.set_xticks([k * 3 + 0.4 for k in range(n_methods)], 
                          method_labels, 
                          fontsize=12,
                          )
            ax.set_xlabel('Prediction Methods', fontsize=14, fontweight='bold')
            ax.set_ylabel('r', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)

        plt.suptitle('Within metric prediction', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig_path = self.graph_path / 'PredWithinMetrics.png'
        plt.savefig(fig_path, dpi=384)
        plt.close()
        print(fig_path)

    def plot_btw_var(self):
        """ plot results for between metric prediction """

        # init
        data = self.pred_results['across']
        sources = ['subj', 'inst']
        methods = ['rand', 'avg', 'corr']
        n_sources = len(sources)
        n_methods = len(methods)
        n_bs, n_met_pairs, n_subjs = data['subj']['rand'].shape

        # setup plot data
        trans_data = np.empty((n_sources, n_methods, n_met_pairs, n_subjs))
        for i, sor in enumerate(sources):
            for j, met in enumerate(methods):
                sub_data = data[sor][met]
                # average across bs / average across condition
                sub_data = np.nanmean(sub_data, axis=0)  
                trans_data[i, j] = sub_data
        trans_data = rearrange(trans_data, 's m met subj -> met s m subj')
        trans_data = np.abs(trans_data)

        # plot
        plt.clf()
        fig, axs = plt.subplots(n_met_pairs // 2, 2, figsize=(6.666, 4 * (n_met_pairs // 2)))
        axs = axs.flatten()
        colors = plt.cm.get_cmap('Dark2', 8)
        labels = ['Predict from Subject', 'Predict from Instance']
        method_labels = ['Rand', 'Avg', 'Corr']

        met_pairs = list(permutations(self.map_var, 2))
        for i, (met_a, met_b) in enumerate(met_pairs):
            ax = axs[i]
            for j, sor in enumerate(sources):
                for k, method in enumerate(methods):
                    x_pos = j * 0.8 + k * 3
                    ax.bar(x_pos,
                           np.mean(trans_data[i,j,k,:]),
                           yerr = sem(trans_data[i,j,k,:]),
                           color = colors(j),
                           alpha = 0.5,
                           label = labels[j] if k == 0 else None,
                           )
                    ax.scatter([x_pos - 0.25 for _ in range(n_subjs)],
                               trans_data[i,j,k,:],
                               color = colors(j),
                               s = 5,
                              )

            ax.set_ylim(-0.1, 1)
            ax.set_title(f"{met_a} predict {met_b}", fontsize=14)
            ax.set_xticks([k * 3 + 0.4 for k in range(n_methods)], 
                          method_labels, 
                          fontsize=12,
                          )
            ax.set_xlabel('Prediction Methods', fontsize=14, fontweight='bold')
            ax.set_ylabel('r', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)

        plt.suptitle('Between metric prediction', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig_path = self.graph_path / 'PredBtwMetrics.png'
        plt.savefig(fig_path, dpi=384)
        plt.close()
        print(fig_path)
