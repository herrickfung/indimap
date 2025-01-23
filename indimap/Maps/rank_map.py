from itertools import combinations
from pathlib import Path
from math import comb
import einops
import numpy as np

from .corr_map import CorrMap
from .util.stat_func import stat_func
from .util import map_func as map_func

class RankMap:
    def __init__(self, config):

        self.config = config
        self.corr_map = CorrMap(self.config)

        default_config = {
            'map_options': {
                'CorrMap': True,
                'RankMap': True,
                'TopMap': True,
            },
            'bootstrap_iterations': 10,
            'bootstrap_seed': 42,
            'load_exists': False,
            'output_path': 'IndiMap_Result',
        }

        self.config = {**default_config, **config}

        self.human = self.config.get('subj_data')
        self.model = self.config.get('inst_data')
        self.human_iden = self.config.get('subj_column_name', 'subj')
        self.model_iden = self.config.get('inst_column_name', 'inst')
        self.map_var = self.config.get('map_variables', ['acc', 'conf'])
        self.map_tgt = self.config.get('map_together')
        self.map_sep = self.config.get('map_separate', [None])

        self.n_bs = self.config['bootstrap_iterations']
        self.bs_seed = self.config['bootstrap_seed']
        self.output_path = Path(self.config['output_path'])

        self.rank_results = None

    def load_all(self):
        """ Loads precomputed results from a file """
        loaded = np.load(self.output_path / 'RankMap_results.npz', allow_pickle=True)
        self.rank_results = loaded['rank_results'].item()

    def save_all(self):
        """ Save results to a file """
        output = {
            'rank_results': self.rank_results,
        }
        output_path = self.output_path / 'RankMap_results.npz'
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
        self.results = {
            map_type: self.do_rank_analysis(map_type) for map_type in map_dicts
            }


    def do_rank_analysis(self, map_type):
        """Main analysis pipeline on the count and correlations of top performers"""
        corr_path = self.output_path / 'CorrMap_results.npz'
        if corr_path.is_file():
            self.load_map_from_corr(self.output_path)
        else:
            self.compute_corr_map()

        btw_split = self.optim_sorcd_btw_split(self.corr_map.corr_maps[map_type])
        if len(self.corr_map.map_var) > 1:
            btw_var = self.compute_btw_var(self.corr_map.corr_maps[map_type])
        else:
            btw_var = None

        return {
            'btw_split': btw_split,
            'btw_var': btw_var,
        }


    def compute_btw_var(self, data):
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
    def optim_sorcd_btw_split(data):
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
    def sum_of_ranked_corr_diff(data):
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
