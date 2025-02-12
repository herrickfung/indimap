"""
contains all function related to mapping and analyses exclusively on top model
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import combinations
from scipy.stats import sem
import numpy as np
import einops

from .corr_map import CorrMap
from .util import stat_func, map_func
rcParams['font.family'] = 'CMU Sans Serif'


class TopMap:
    def __init__(self, config):
        self.config = config
        self.corr_map = CorrMap(self.config)
        self.top_maps = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
        }
        self.top_ct = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
        }
        self.top_corr = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
        }
        self.top_results = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
        }

    def check_exist(self, path):
        # check whether the map exist
        file_path = path / 'TopMap_results.npz'
        return file_path.exists()

    def load_all(self, path):
        """ Loads precomputed results from a file """
        loaded = np.load(path / 'TopMap_results.npz', allow_pickle=True)
        self.top_maps = loaded['top_maps'].item()
        self.top_ct = loaded['top_ct'].item()
        self.top_corr = loaded['top_corr'].item()
        self.top_results = loaded['top_results'].item()

    def save_all(self, path):
        """ Save results to a file """
        output = {
            'top_maps': self.top_maps,
            'top_ct': self.top_ct,
            'top_corr': self.top_corr,
            'top_results': self.top_results,
        }
        output_path = path / 'TopMap_results.npz'
        np.savez(output_path, **output)

    def load_map_from_corr(self, path):
        """ Loads precomputed results from CorrMap class"""
        self.corr_map.load_map()

    def compute_corr_map(self):
        """ Computes and saves in CorrMap"""
        self.corr_map.compute_corr_maps()
        self.corr_map.save_map()

    def compute_top_analysis(self):
        """Performs the top-level analysis for all subtypes of Maps"""

        name_dicts = [
            'subj_to_inst',
            'subj_to_subj',
            'inst_to_inst'
        ]
        for name in name_dicts:
            self.top_maps[name] = self.get_top(self.corr_map.corr_maps[name])
            self.top_ct[name], self.top_corr[name] = self.get_counts_and_corr(self.top_maps[name])
            self.top_results[name] = self.do_top_analysis(name)

    def do_top_analysis(self, key) -> dict:
        """Main analysis pipeline on the count and correlations of top performers"""
        ct_btw_split_results = self.corr_btw_split(self.top_ct[key])
        corr_btw_split_results = self.corr_btw_split(self.top_corr[key], True)

        if len(self.corr_map.map_var) > 1:
            ct_btw_var_results = self.corr_btw_var(self.top_ct[key])
            corr_btw_var_results = self.corr_btw_var(self.top_corr[key], True)
        else:
            ct_btw_var_results = None
            corr_btw_var_results = None

        return {
            "ct_btw_split": ct_btw_split_results,
            "corr_btw_split": corr_btw_split_results,
            "ct_btw_var": ct_btw_var_results,
            "corr_btw_var": corr_btw_var_results
        }

    def plot_top_average(self) -> None:
        """ Plot the average top analysis results. Would indicate top mapping accuracy """

        plt.clf()
        plt.figure(figsize=(6, 4))
        colors = plt.cm.get_cmap('Dark2', 8)
        n_metrics = len(self.corr_map.map_var)
        map_types = ['subj_to_inst', 'subj_to_subj', 'inst_to_inst']
        map_labels = ['Subj to Inst', 'Subj to Subj', 'Inst to Inst']

        for i, map_type in enumerate(map_types):
            for j in range(n_metrics):
                x_pos = j * 3 + i * 0.8
                plt.bar(x_pos, 
                        self.top_corr[map_type][:,:,j,:].mean(),
                        yerr = sem(self.top_corr[map_type][:,:,j,:], axis = (0,1,2)),
                        color = colors(i), alpha = 0.5, label = map_labels[i] if j == 0 else None
                        )

        plt.xticks([0.8 + i * 3 for i in range(n_metrics)], self.corr_map.map_var, fontsize=12)
        plt.xlabel('Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('r', fontsize=12, fontweight='bold')
        plt.legend()
        plt.title('Average of Correlation Matrices (Top performer)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_path = f'{self.corr_map.graph_path}/TopAvg.png'
        plt.savefig(fig_path, dpi=384)
        plt.close()

    def plot_btw_split(self) -> None:
        """
        Plot the top analysis results between bootstrap splits.
        """

        def plot_data(trans_data, title, filename):
            plt.clf()
            plt.figure(figsize=(6, 4))
            colors = plt.cm.get_cmap('Dark2', 8)
            map_labels = ['Subj to Inst', 'Subj to Subj', 'Inst to Inst']

            for i, map in enumerate(map_types):
                for j in range(n_metrics):
                    x_pos = j * 3 + i * 0.8
                    plt.bar(x_pos, 
                            trans_data[j, i, :].mean(),
                            yerr=sem(trans_data[j, i, :]),
                            color=colors(i),
                            alpha=0.5,
                            label=map_labels[i] if j == 0 else None
                            )

            plt.ylim(0, 1.1)
            plt.xticks([i * len(map_types) + 0.8  for i in range(n_metrics)], self.corr_map.map_var, fontsize=12)
            plt.xlabel('Metrics', fontsize=12, fontweight='bold')
            plt.ylabel('r', fontsize=12, fontweight='bold')
            plt.legend()
            plt.title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig_path = f'{self.corr_map.graph_path}/{filename}'
            plt.savefig(fig_path, dpi=384)
            plt.close()

        n_metrics = len(self.corr_map.map_var)
        map_types = ['subj_to_inst', 'subj_to_subj', 'inst_to_inst']

        # Plot for count between splits
        trans_data = np.empty(shape=(n_metrics, len(map_types), self.corr_map.n_bs))
        for i, map in enumerate(map_types):
            for j in range(n_metrics):
                trans_data[j, i, :] = self.top_results[map]['ct_btw_split'][:, j]
        plot_data(trans_data, 'Correlation between image splits (Best Instance Count)', 'TopCountBtwSplit.png')

        # Plot for correlation between splits
        trans_data = np.empty(shape=(n_metrics, len(map_types), self.corr_map.n_bs))
        for i, map in enumerate(map_types):
            for j in range(n_metrics):
                trans_data[j, i, :] = self.top_results[map]['corr_btw_split'][:, j]
        plot_data(trans_data, 'Correlation between image splits (Best Instance Correlation)', 'TopCorrBtwSplit.png')

    def plot_btw_var(self) -> None:

        def plot_data(trans_data, title, filename):
            plt.clf()
            plt.figure(figsize=(6, 4))
            colors = plt.cm.get_cmap('Dark2', 8)
            map_labels = ['Subj to Inst', 'Subj to Subj', 'Inst to Inst']
            xticks_labels = [ f'{self.corr_map.map_var[i]}-{self.corr_map.map_var[j]}' for i,j in metric_pairs]

            for i, map in enumerate(map_types):
                for j in range(n_metric_pair):
                    x_pos = j * len(map_types) + i * 0.8
                    plt.bar(x_pos, 
                            trans_data[j, i, :].mean(),
                            yerr=sem(trans_data[j, i, :]),
                            color=colors(i),
                            alpha=0.5,
                            label=map_labels[i] if j == 0 else None
                            )

            plt.ylim(0, 1.1)
            plt.xticks([i * len(map_types) + 0.8  for i in range(n_metric_pair)], xticks_labels, fontsize=12)
            plt.xlabel('Pairs of Metric', fontsize=12, fontweight='bold')
            plt.ylabel('r', fontsize=12, fontweight='bold')
            plt.legend()
            plt.title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig_path = f'{self.corr_map.graph_path}/{filename}'
            plt.savefig(fig_path, dpi=384)
            plt.close()

        metric_pairs = list(combinations(range(len(self.corr_map.map_var)), 2))
        n_metric_pair = len(metric_pairs)
        map_types = ['subj_to_inst', 'subj_to_subj', 'inst_to_inst']

        # Plot for count between splits
        trans_data = np.empty(shape=(n_metric_pair, len(map_types), self.corr_map.n_bs))
        for i, map in enumerate(map_types):
            for j in range(n_metric_pair):
                trans_data[j, i, :] = self.top_results[map]['ct_btw_var'][:, j]
        plot_data(trans_data, 'Correlation between metrics (Best Instance Count)', 'TopCountBtwMetrics.png')

        # Plot for correlation between splits
        trans_data = np.empty(shape=(n_metric_pair, len(map_types), self.corr_map.n_bs))
        for i, map in enumerate(map_types):
            for j in range(n_metric_pair):
                trans_data[j, i, :] = self.top_results[map]['corr_btw_var'][:, j]
        plot_data(trans_data, 'Correlation between metrics (Best Instance Correlation)', 'TopCorrBtwMetrics.png')

    def plot_all(self) -> None:
        """Plot all top analysis results."""
        self.plot_top_average()
        self.plot_btw_split()
        self.plot_btw_var()

    @staticmethod
    def get_top(data) -> np.ndarray:
        """
        get the top performer of the input data, output shape unchanged
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        """

        result = np.empty(shape=data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    # add a small constant to prevent nan
                    data_slice = data[i, j, k, ...]
                    data_slice = np.where(np.isnan(data_slice), 1e-10, data_slice)
                    result[i,j,k,:] = map_func.retain_max_per_row_in_mat(data_slice)
        return result

    @staticmethod
    def get_counts_and_corr(data) -> tuple:
        """
        get the count of top performer for each instance,
        and the best correlation value for each subject
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        """

        count_results = np.empty(shape=(data.shape[0], data.shape[1],
                                        data.shape[2], data.shape[4],
                                        ))
        corr_results = np.empty(shape=(data.shape[0], data.shape[1],
                                       data.shape[2], data.shape[3],
                                       ))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    data_slice = data[i,j,k,...]
                    non_nan = ~np.isnan(data_slice)

                    # get count
                    count = np.sum(non_nan, axis = 0)
                    count_results[i,j,k,:] = count

                    # get best correlation value
                    corr_results[i,j,k,:] = data_slice[non_nan]

        return count_results, corr_results

    @staticmethod
    def corr_btw_split(data, is_pearson=False) -> np.ndarray:
        """
        correlate count/correlation of subj/inst between split half bootstrap
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap, split, metrics, count/corr].
        is_pearson: do z transformation on data if True
        """

        if is_pearson:
            data = stat_func.r2z(data, 'pearson')
        data = einops.rearrange(data, 'boot split met value -> boot met split value')
        results = np.empty(shape = (data.shape[0], data.shape[1]))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_slice = data[i, j]
                results[i,j] = np.corrcoef(data_slice)[0,1]
        return results

    @staticmethod
    def corr_btw_var(data, is_pearson=False) -> np.ndarray:
        """
        correlate count/correlation of subj/inst between metrics, Acc, RT, Conf, etc.
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap, split, metrics, count/corr].
        is_pearson: do z transformation on data if True
        output: array [bootstrap, metric_pair]
        """

        if is_pearson:
            data = stat_func.r2z(data, 'pearson')

        unique_var_pairs = list(combinations(range(data.shape[2]), 2))

        results = np.empty(shape = (data.shape[0], data.shape[1],
                                    len(unique_var_pairs),
                                    ))

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k, (a,b) in enumerate(unique_var_pairs):
                    data_a = data[i,j,a]
                    data_b = data[i,j,b]
                    results[i,j,k] = np.corrcoef(data_a, data_b)[0,1]

        # average across splits
        results = stat_func.r2z(results, 'pearson')
        results = np.mean(results, axis=1)
        results = stat_func.z2r(results, 'pearson')

        return results

