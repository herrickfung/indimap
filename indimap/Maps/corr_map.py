'''
contains all function related to mapping and analyses by correlation
'''

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.manifold import MDS
from pathlib import Path
from tqdm import tqdm
import einops
import numpy as np
import pandas as pd
import pickle
import time
import torch

from .util.stat_func import stat_func
from .util import map_func as map_func


class CorrMap:
    def __init__(self, config):

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
        self.human_iden = self.config.get('subj_column_name')
        self.model_iden = self.config.get('inst_column_name')
        self.map_var = self.config.get('map_variables')
        self.map_tgt = self.config.get('map_together')
        self.map_sep = self.config.get('map_separate')

        self.n_bs = self.config['bootstrap_iterations']
        self.bs_seed = self.config['bootstrap_seed']
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


    def compute_corr_maps(self):
        human_arr = map_func.convert_to_array(self.human, self.human_iden,
                                              self.map_var, self.map_tgt,
                                              self.map_sep
                                              )
        model_arr = map_func.convert_to_array(self.model, self.model_iden,
                                              self.map_var, self.map_tgt,
                                              self.map_sep
                                              )
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

    def compute_corr_analysis(self):
        map_dicts = ['subj_to_inst',
                     'subj_to_subj',
                     'inst_to_inst',
                     ]

        self.corr_results = {
            map_type: self.do_corr_analysis(map_type) for map_type in map_dicts
        }


    def compute_mds(self):
        data = self.corr_maps['combined_map']
        results = np.empty(shape=(data.shape[0], data.shape[1], data.shape[2], 2))

        # initalize MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        for ds in range(data.shape[0]):
            for metric in range(data.shape[1]):
                input_data = 1 - data[ds, metric]
                results[ds, metric] = mds.fit_transform(input_data)
        self.mds_results['mds_results'] = results


    def do_corr_analysis(self, key):
        data = stat_func.r2z(self.corr_maps[key], 'pearson')

        # subj_btw_split_results = self.corr_btw_split(data, 3)
        # inst_btw_split_results = self.corr_btw_split(data, 4)
        # subj_to_group_btw_split_results = self.corr_to_group_btw_split(data, 3)
        # inst_to_group_btw_split_results = self.corr_to_group_btw_split(data, 4)

        # ZZZ start from here
        if len(self.map_var) > 1:
            subj_btw_variable_results = self.corr_subj_btw_variable(data)
            subj_to_group_variable_results = self.corr_subj_to_group_variable(data)
        else:
            subj_btw_variable_results = None
            subj_to_group_variable_results = None

        # package results
        results = {
            "subj_btw_splits" : subj_btw_split_results,
            "inst_btw_splits" : inst_btw_split_results,
            "subj_to_group_splits" : subj_to_group_btw_split_results,
            "inst_to_group_splits" : inst_to_group_btw_split_results,
            "subj_btw_variable" : subj_btw_variable_results,
            "subj_to_group_btw_variable" : subj_to_group_variable_results,
        }
        return results


    def load_all(self):
        loaded = np.load(self.output_path / 'CorrMap_results.npz', allow_pickle=True)
        self.corr_maps = loaded['corr_maps'].item()
        self.corr_results = loaded['corr_results'].item()


    def save_all(self):
        output = {
            'corr_maps': self.corr_maps,
            'corr_results': self.corr_results,
        }
        output_path = self.output_path / 'CorrMap_results.npz'
        np.savez(output_path, **output)


    @staticmethod
    def corr_btw_split(data, axis):
        """
        Compute the correlation between two bootstrap_splits
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        axis (str): Specifies axis to compute correlation on.
        ---------------------------------------------------------------------------
        """
        results = np.empty(shape=(data.shape[0], data.shape[2], data.shape[axis]))

        if axis == 3:
            data = einops.rearrange(data, 'boot split met subj inst -> boot met subj inst split')
        elif axis == 4:
            data = einops.rearrange(data, 'boot split met subj inst -> boot met inst subj split')

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    data_slice = data[i, j, k, :, :]
                    results[i,j,k] = np.corrcoef(data_slice, rowvar=False)[0,1]
        return results



    @staticmethod
    def corr_to_group_btw_split(data, axis):
        """
        Compute the correlation between subject and n - 1 subjects between two bootstrap splits
        ---------------------------------------------------------------------------
        Parameters:
        ---------------------------------------------------------------------------
        data (np.ndarray): The input data array with shape
        [bootstrap split, split-half, metrics, subj, inst].
        axis (str): Specifies axis to compute correlation on.
        ---------------------------------------------------------------------------
        """
        results = np.empty(shape=(data.shape[0],    # number of bootstrap splits
                                  data.shape[1],    # split half
                                  data.shape[2],    # metrics
                                  data.shape[axis], # subj/inst
                                  ))
        if axis == 4:
            data = einops.rearrange(data, 'boot split met subj inst -> boot split met inst subj')

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    data_slice = data[i, j, k, :, :]
                    full_mat = map_func.compute_full_corr_matrix(data_slice, data_slice)
                    full_mat = stat_func.r2z(full_mat, 'pearson')
                    np.fill_diagonal(full_mat, np.nan)
                    result = np.nanmean(full_mat, axis = 0)
                    result = stat_func.z2r(result, 'pearson')
                    results[i,j,k,:] = result
        return results


