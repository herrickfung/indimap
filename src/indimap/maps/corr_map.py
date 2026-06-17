'''
contains all functions related to mapping and analyses by correlation
'''

import matplotlib.pyplot as plt
from matplotlib import rcParams
from itertools import combinations
from scipy.stats import sem
from pathlib import Path
import einops
import numpy as np

from indimap.util import stat_func, map_func

rcParams['font.family'] = 'CMU Sans Serif'


class CorrMap:
    def __init__(self, config: dict):
        """Initializes the CorrMap class."""

        self.config = config
        self.human = self.config.get('subj_data')
        self.model = self.config.get('inst_data')
        self.human_iden = self.config.get('subj_column_name')
        self.model_iden = self.config.get('inst_column_name')
        self.stim_iden = self.config.get('stim_column_name')
        self.map_var = list(self.config.get('map_variables'))
        self.map_tgt = self.config.get('map_together')
        self.map_sep = self.config.get('map_separate')
        self.map_confusion = self.config.get('map_confusion')
        self.map_category = self.config.get('map_category')
        self.n_bs = self.config.get('bootstrap_iterations')
        self.bs_seed = self.config.get('bootstrap_seed')
        self.output_path = Path(self.config['output_path'])
        self.graph_path = Path(self.config['graph_path'])

        if self.map_confusion and 'confuse_mat' not in self.map_var:
            self.map_var.append('confuse_mat')

        self.corr_maps = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
        }
        self.cat_corr_maps = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
        }
        self.corr_results = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
        }
        self.cat_corr_results = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
        }

    def check_exist(self):
        """ Check whether the file exist """
        file_path = self.output_path / 'CorrMap_results.npz'
        return file_path.exists()

    def load_all(self):
        """ Loads precomputed results from a file """
        loaded = np.load(self.output_path / 'CorrMap_results.npz', allow_pickle=True)
        self.corr_maps = loaded['corr_maps'].item()
        self.cat_corr_maps = loaded['cat_corr_maps'].item()
        self.corr_results = loaded['corr_results'].item()
        self.cat_corr_results = loaded['cat_corr_results'].item()

    def save_all(self):
        """ Save results to a file """
        output = {
            'corr_maps': self.corr_maps,
            'corr_results': self.corr_results,
            'cat_corr_maps': self.cat_corr_maps,
            'cat_corr_results': self.cat_corr_results,
        }
        output_path = self.output_path / 'CorrMap_results.npz'
        np.savez(output_path, **output)

    def load_map(self):
        """ Loads maps for other classes"""
        loaded = np.load(self.output_path / 'CorrMap_results.npz', allow_pickle=True)
        self.corr_maps = loaded['corr_maps'].item()

    def save_map(self):
        """ Saves maps for other classes """
        output = {
            'corr_maps': self.corr_maps,
        }
        output_path = self.output_path / 'CorrMap_results.npz'
        np.savez(output_path, **output)

    def compute_corr_maps(self, sep_conds: bool = False) -> None:
        """pipeline from raw data to correlation maps."""
        human_arr = map_func.convert_to_array(self.human, self.human_iden,
                                              self.map_var, self.map_tgt,
                                              self.map_sep, self.map_confusion
                                              )
        model_arr = map_func.convert_to_array(self.model, self.model_iden,
                                              self.map_var, self.map_tgt,
                                              self.map_sep, self.map_confusion
                                              )
        model_arr = map_func.check_for_extreme(human_arr, model_arr)

        self.corr_maps = {
            'subj_to_inst': map_func.mapping_matrix(human_arr, model_arr, 
                                                    self.map_var, same=False,
                                                    sep_conds = sep_conds,
                                                    n_bs = self.n_bs, seed=self.bs_seed
                                                    ),
            'subj_to_subj': map_func.mapping_matrix(human_arr, human_arr, 
                                                    self.map_var, same=True,
                                                    sep_conds = sep_conds,
                                                    n_bs = self.n_bs, seed=self.bs_seed
                                                    ),
            'inst_to_inst': map_func.mapping_matrix(model_arr, model_arr, 
                                                    self.map_var, same=True,
                                                    sep_conds = sep_conds,
                                                    n_bs = self.n_bs, seed=self.bs_seed
                                                    ),
        }

        if self.map_category:
            cat_human_arr = map_func.convert_to_array_per_category(self.human, self.human_iden,
                                                                   self.stim_iden,
                                                                   self.map_var, self.map_tgt,
                                                                   self.map_sep,
                                                                   )
            cat_model_arr = map_func.convert_to_array_per_category(self.model, self.model_iden,
                                                                   self.stim_iden,
                                                                   self.map_var, self.map_tgt,
                                                                   self.map_sep,
                                                                   )

            self.cat_corr_maps = {
                'subj_to_inst': map_func.across_category_mapping_matrix(cat_human_arr, cat_model_arr, 
                                                                        same=False, sep_conds = sep_conds,
                                                                        n_bs = self.n_bs, seed=self.bs_seed
                                                                        ),
                'subj_to_subj': map_func.across_category_mapping_matrix(cat_human_arr, cat_human_arr,
                                                                        same=True, sep_conds = sep_conds,
                                                                        n_bs = self.n_bs, seed=self.bs_seed
                                                                        ),
                # 'inst_to_inst':  unsolved bug likely due to nan correlation results from some pairs
            }

    def compute_corr_analysis(self) -> None:
        """Perform correlation analyses on all correlation maps."""
        map_dicts = [
            'subj_to_inst',
            'subj_to_subj',
            'inst_to_inst',
        ]
        self.corr_results = {
            map_type: self.do_corr_analysis(map_type) for map_type in map_dicts
        }

        if self.map_category:
            cat_map_dicts = [
                'subj_to_inst',
                'subj_to_subj',
            ]
            self.cat_corr_results = {
                map_type: self.do_corr_analysis(map_type) for map_type in cat_map_dicts
            }

    def do_corr_analysis(self, key: str) -> dict:
        """pipeline for performing all analysis in the correlation map matrix"""
        # convert to z scores before correlating again in all below
        data = stat_func.r2z(self.corr_maps[key], 'pearson')

        # compute split half correlations
        subj_btw_split_results, subj_to_group_btw_split_results = self.corr_btw_split(data, 3)
        inst_btw_split_results, inst_to_group_btw_split_results = self.corr_btw_split(data, 4)

        # compute between metric correlations if more than one metric is present
        if len(self.map_var) > 1:
            subj_btw_var_results, subj_to_group_btw_var_results = self.corr_btw_var(data, axis = 3)
            inst_btw_var_results, inst_to_group_btw_var_results = self.corr_btw_var(data, axis = 4)
        else:
            subj_btw_var_results = None
            inst_btw_var_results = None
            subj_to_group_btw_var_results = None
            inst_to_group_btw_var_results = None

        return {
            "subj_btw_split": subj_btw_split_results,
            "subj_gp_btw_split": subj_to_group_btw_split_results,
            "inst_btw_split": inst_btw_split_results,
            "inst_gp_btw_split": inst_to_group_btw_split_results,

            "subj_btw_var" : subj_btw_var_results,
            "subj_gp_btw_var" : subj_to_group_btw_var_results,
            "inst_btw_var" : inst_btw_var_results,
            "inst_gp_btw_var" : inst_to_group_btw_var_results,
        }
    
    def plot_map_average(self) -> None:
        """Plot the average of the map. Would indicate the overall mapping accuracy"""

        plt.clf()
        plt.figure(figsize=(6, 4))
        colors = plt.cm.get_cmap('Dark2', 8)
        n_metrics = len(self.map_var)
        map_types = ['subj_to_inst', 'subj_to_subj', 'inst_to_inst']
        map_labels = ['Subj to Inst', 'Subj to Subj', 'Inst to Inst']

        for i, map_type in enumerate(map_types):
            for j in range(n_metrics):
                x_pos = j * 3 + i * 0.8
                plt.bar(x_pos, 
                        np.nanmean(self.corr_maps[map_type][:,:,j,:,:], axis = (0,1,2,3)),
                        yerr = sem(self.corr_maps[map_type][:,:,j,:,:], axis = (0,1,2,3)),
                        color = colors(i), alpha = 0.5, label = map_labels[i] if j == 0 else None
                        )

        plt.xticks([0.8 + i * 3 for i in range(n_metrics)], self.map_var, fontsize=12)
        plt.xlabel('Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('r', fontsize=12, fontweight='bold')
        plt.legend()
        plt.title('Average of Correlation Matrices', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_path = self.graph_path / 'CorrAvg.png'
        plt.savefig(fig_path, dpi=384)
        plt.close()
        print(fig_path)
    
    def plot_btw_split(self) -> None:
        """ Plot the correlation between bootstrap splits of images """

        n_bars = 2
        n_maps = 3
        n_metrics = len(self.map_var)
        n_subjs = self.human[self.human_iden].nunique()

        # transform data and average across bootstrap splits
        trans_data = np.empty(shape=(n_metrics, n_maps, n_bars, n_subjs))
        map_type = ['subj_to_inst', 'subj_to_subj', 'inst_to_inst']
        result_type = ['subj_btw_split', 'subj_gp_btw_split']
        for i, map in enumerate(map_type):
            for j, result in enumerate(result_type):
                trans_data[:, i, j, :] = np.mean(self.corr_results[map][result], axis = 0)   

        # plot here
        plt.clf()
        plt.figure(figsize=(8, 6))
        colors = plt.cm.get_cmap('Dark2', 8)
        map_labels = ['Subj to Inst', 'Scrambled Subj to Inst', 
                     'Subj to Subj', 'Scrambled Subj to Subj',
                     'Inst to Inst', 'Scrambled Inst to Inst'
                     ]

        for i in range(n_metrics):
            for j in range(n_maps):
                for k in range(n_bars):
                    x_pos = i * 10 + j * 2.5 + k * 0.8

                    plt.bar(x_pos,
                            np.mean(trans_data[i,j,k,:], axis = 0),
                            yerr = sem(trans_data[i,j,k,:], axis = 0),
                            color = colors(j * 2 + k),
                            alpha = 0.5,
                            label = map_labels[j * 2 + k] if i == 0 else None,
                            )

                    plt.scatter([x_pos-0.25 for _ in range(n_subjs)],
                                trans_data[i,j,k,:],
                                color = colors(j * 2 + k),
                                s = 5,
                                )

        plt.xticks([i * 10 + 2.5 for i in range(n_metrics)], self.map_var, fontsize=14)
        plt.ylim(-0.6, 1.1)
        plt.xlabel('Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('r', fontsize=14, fontweight='bold')
        plt.legend()
        plt.title('Correlation between bootstrap splits of images', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig_path = self.graph_path / 'CorrBtwSplit.png'
        plt.savefig(fig_path, dpi=384)
        plt.close()
        print(fig_path)

    def plot_btw_var(self) -> None:
        """ plot the correlation between metrics """

        n_bars = 2
        n_maps = 3
        metric_pairs = list(combinations(range(len(self.map_var)), 2))
        n_metric_pairs = len(metric_pairs)
        n_subjs = self.human[self.human_iden].nunique()

        # transform data and average across bootstrap splits
        trans_data = np.empty(shape=(n_metric_pairs, n_maps, n_bars, n_subjs))
        map_type = ['subj_to_inst', 'subj_to_subj', 'inst_to_inst']
        result_type = ['subj_btw_var', 'subj_gp_btw_var']
        for i, map in enumerate(map_type):
            for j, result in enumerate(result_type):
                trans_data[:, i, j, :] = np.mean(self.corr_results[map][result], axis = 0)

        # plot here
        plt.clf()
        plt.figure(figsize=(8, 6))
        colors = plt.cm.get_cmap('Dark2', 8)
        map_labels = ['Subj to Inst', 'Scrambled Subj to Inst', 
                     'Subj to Subj', 'Scrambled Subj to Subj',
                     'Inst to Inst', 'Scrambled Inst to Inst'
                     ]
        xticks_labels = [ f'{self.map_var[i]}-{self.map_var[j]}' for i,j in metric_pairs]

        for i in range(n_metric_pairs):
            for j in range(n_maps):
                for k in range(n_bars):
                    x_pos = i * 10 + j * 2.5 + k * 0.8

                    plt.bar(x_pos,
                            np.mean(trans_data[i,j,k,:], axis = 0),
                            yerr = sem(trans_data[i,j,k,:], axis = 0),
                            color = colors(j * 2 + k),
                            alpha = 0.5,
                            label = map_labels[j * 2 + k] if i == 0 else None,
                            )

                    plt.scatter([x_pos-0.25 for _ in range(n_subjs)],
                                trans_data[i,j,k,:],
                                color = colors(j * 2 + k),
                                s = 5,
                                )

        plt.xticks([i * 10 + 2.5 for i in range(n_metric_pairs)], xticks_labels, fontsize=14)
        plt.ylim(-0.6, 1.1)
        plt.xlabel('Pairs of Metric', fontsize=14, fontweight='bold')
        plt.ylabel('r', fontsize=14, fontweight='bold')
        plt.legend()
        plt.title('Correlation between metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig_path = self.graph_path / 'CorrBtwMetrics.png'
        plt.savefig(fig_path, dpi=384)
        plt.close()
        print(fig_path)

    @staticmethod
    def corr_btw_split(data: np.ndarray, axis: int) -> tuple:
        """
        Compute the correlation between bootstrap splits of image data
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        axis (int): Specifies axis to compute correlation on.
        """

        unique_spt_pairs = list(combinations(range(data.shape[1]), 2))
        within_results = np.empty(shape=(data.shape[0],     # number of bootstrap splits
                                  data.shape[2],            # metric
                                  len(unique_spt_pairs),    # unique split pair
                                  data.shape[axis],         # within subj correlation results
                                  ))
        between_results = np.empty(shape=(data.shape[0],     # number of bootstrap splits
                                   data.shape[2],            # metric
                                   len(unique_spt_pairs),    # unique split pair
                                   data.shape[axis],         # average of n-1 correlation results
                                   ))

        if axis == 3:
            data = einops.rearrange(data, 'boot split met subj inst -> boot met split subj inst')
        elif axis == 4:
            data = einops.rearrange(data, 'boot split met subj inst -> boot met split inst subj')

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k, (a,b) in enumerate(unique_spt_pairs):
                    data_a = data[i,j,a,:,:]
                    data_b = data[i,j,b,:,:]
                    full_mat = map_func.compute_full_corr_matrix(data_a, data_b)

                    within_results[i,j,k,:] = np.diagonal(full_mat)

                    full_mat = stat_func.r2z(full_mat, 'pearson')
                    np.fill_diagonal(full_mat, np.nan)
                    result = np.nanmean(full_mat, axis = 0)
                    between_results[i,j,k,:] = result


        within_results = stat_func.r2z(within_results, 'pearson')
        within_results = np.nanmean(within_results, axis = 2)
        within_results = stat_func.z2r(within_results, 'pearson')

        between_results = np.nanmean(between_results, axis = 2)
        between_results = stat_func.z2r(between_results, 'pearson')

        return within_results, between_results


    @staticmethod
    def corr_btw_var(data: np.ndarray, axis: int) -> tuple:
        """
        Compute the correlation between metrics
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        axis (int): Specifies axis to compute correlation on.
        """

        unique_var_pairs = list(combinations(range(data.shape[2]), 2))

        within_results = np.empty(shape=(data.shape[0],     # number of bootstrap splits
                                  data.shape[1],            # split half
                                  len(unique_var_pairs),    # unique metrics pair
                                  data.shape[axis],         # within subj correlation results
                                  ))
        between_results = np.empty(shape=(data.shape[0],    # number of bootstrap splits
                                   data.shape[1],           # split half
                                   len(unique_var_pairs),   # unique metrics pair
                                   data.shape[axis],        # average of n-1 correlation results
                                   ))

        if axis == 3:
            data = einops.rearrange(data, 'boot split met subj inst -> boot split met subj inst')
        elif axis == 4:
            data = einops.rearrange(data, 'boot split met subj inst -> boot split met inst subj')

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k, (a,b) in enumerate(unique_var_pairs):
                    data_a = data[i,j,a,:,:]
                    data_b = data[i,j,b,:,:]
                    full_mat = map_func.compute_full_corr_matrix(data_a, data_b)

                    within_results[i,j,k,:] = np.diagonal(full_mat)

                    full_mat = stat_func.r2z(full_mat, 'pearson')
                    np.fill_diagonal(full_mat, np.nan)
                    result = np.nanmean(full_mat, axis = 0)
                    between_results[i,j,k,:] = result

        within_results = stat_func.r2z(within_results, 'pearson')
        within_results = np.nanmean(within_results, axis = 1)
        within_results = stat_func.z2r(within_results, 'pearson')

        between_results = np.nanmean(between_results, axis = 1)
        between_results = stat_func.z2r(between_results, 'pearson')

        return within_results, between_results

