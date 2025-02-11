'''
contains all function related to mapping and analyses by correlation
'''

from itertools import combinations
from pathlib import Path
from math import comb
import einops
import numpy as np

from .corr_map import CorrMap
from .util import stat_func, map_func


class RankMap:
    def __init__(self, config):
        self.config = config
        self.corr_map = CorrMap(self.config)
        self.rank_results = None

    def check_exist(self, path):
        """ Check if the file exists """
        file_path = path / 'RankMap_results.npz'
        return file_path.exists()

    def load_all(self, path):
        """ Loads precomputed results from a file """
        loaded = np.load(path / 'RankMap_results.npz', allow_pickle=True)
        self.rank_results = loaded['rank_results'].item()

    def save_all(self, path):
        """ Save results to a file """
        output = {
            'rank_results': self.rank_results,
        }
        output_path = path / 'RankMap_results.npz'
        np.savez(output_path, **output)

    def load_map_from_corr(self, path):
        """ Loads precomputed results from CorrMap class"""
        self.corr_map.load_map()

    def compute_corr_map(self):
        """ Computes and saves in CorrMap"""
        self.corr_map.compute_corr_maps()
        self.corr_map.save_map()

    def compute_rank_analysis(self):
        """Perform rank analyses on all maps."""
        map_dicts = [
            'subj_to_inst',
            'subj_to_subj',
            'inst_to_inst',
        ]
        self.rank_results = {
            map_type: self.do_rank_analysis(map_type) for map_type in map_dicts
            }

    def do_rank_analysis(self, map_type) -> dict:
        """Main analysis pipeline on the count and correlations of top performers"""

        btw_split = self.optim_sorcd_btw_split(self.corr_map.corr_maps[map_type])

        if len(self.corr_map.map_var) > 1:
            btw_var = self.compute_sorcd_btw_var(self.corr_map.corr_maps[map_type])
        else:
            btw_var = None

        return {
            'btw_split': btw_split,
            'btw_var': btw_var,
        }

    def compute_sorcd_btw_var(self, data) -> np.ndarray:
        """
        computing SORCD for between variables
        between split were averaged out
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        output (np.ndarray): The computed SORCD values with shape [bootstrap, pairs_of_metric].
        """

        unique_pairs = list(combinations(range(data.shape[2]), 2))
        list_in_data = [data[:,:,[i,j],:,:] for i, j in unique_pairs]
        results = np.empty(shape=(data.shape[0], len(list_in_data)))
        for i, in_data in enumerate(list_in_data):
            in_data = einops.rearrange(in_data, 'boot split met subj inst -> boot met split subj inst')
            result = self.optim_sorcd_btw_split(in_data)
            result = result.mean(axis = 1)  # average across splits
            results[:,i] = result
        return results

    @staticmethod
    def optim_sorcd_btw_split(data) -> np.ndarray:
        """
        optimized way of computing SORCD,
        refer to the function below for more details
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        output (np.ndarray): The computed SORCD values with shape [bootstrap, metrics].
        """

        n_boots = data.shape[0]
        n_splits = data.shape[1]
        n_metrics = data.shape[2]
        n_pairs = comb(data.shape[-1], 2)
        data_diff = np.empty(shape=(n_boots, n_splits, n_metrics, n_pairs, data.shape[-1]))

        data = np.argsort(-data, axis = 4).argsort(axis=4) + 1
        p1, p2 = np.triu_indices(data.shape[-1], k = 1)
        data_diff = data[:, :, :, p2] - data[:, :, :, p1]
        result = np.prod(data_diff, axis = 1)
        result = np.sum(result, axis=(2,3)) / np.prod(result.shape[2:])
        return result

    @staticmethod
    def sorcd(data):
        """
        original idea of rank analysis SORCD
        (Sum of Ranked Correlation Difference)
        """

        # set up
        n_splits = data.shape[0]
        n_pairs = comb(data.shape[-1], 2)
        data_diff = np.empty(shape=(n_splits, n_pairs, data.shape[-1]))
        # rank matrix, this tells how well each subj1 is fit with subj2
        # negate for descending order, add one for 1-based ranking
        # 1 is best
        data = np.argsort(-data, axis = 2).argsort(axis=2) + 1
        # get index for all pairs
        p1, p2 = np.triu_indices(data.shape[-1], k = 1)
        # compute the difference
        data_diff = data[:, p2] - data[:, p1]
        # multiply dataset 1 to dataset 2, dataset is axis 0
        result = np.prod(data_diff, axis = 0)
        # sum everything, except for the metrics axis
        # sum is normalized by total number of elements summed
        result = np.sum(result) / np.prod(result.shape)
        # the result is a single number for each metric
        # the higher the better, the more consistent is the mapping across dataset
        return result
