'''
contains all functions related to mapping and analyses by correlation
'''

from itertools import combinations
from pathlib import Path
import einops
import numpy as np

from .util import stat_func, map_func


class CorrMap:
    def __init__(self, config):
        """Initializes the CorrMap class."""

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

        self.corr_maps = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
        }
        self.corr_results = {
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
        self.corr_results = loaded['corr_results'].item()

    def save_all(self):
        """ Save results to a file """
        output = {
            'corr_maps': self.corr_maps,
            'corr_results': self.corr_results,
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

    def compute_corr_maps(self) -> None:
        """pipeline from raw data to correlation maps."""
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
        self.corr_maps = {
            'subj_to_inst': map_func.mapping_matrix(human_split, model_split),
            'subj_to_subj': map_func.mapping_matrix(human_split, human_split),
            'inst_to_inst': map_func.mapping_matrix(model_split, model_split),
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

    def do_corr_analysis(self, key) -> dict:
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

    @staticmethod
    def corr_btw_split(data, axis) -> tuple:
        """
        Compute the correlation between bootstrap splits of image data
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        axis (str): Specifies axis to compute correlation on.
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
    def corr_btw_var(data, axis) -> tuple:
        """
        Compute the correlation between metrics
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        axis (str): Specifies axis to compute correlation on.
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

