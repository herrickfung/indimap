'''
contains all function related to mapping and analyses by correlation
'''

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.manifold import MDS
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

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
            'bootstrap_iterations': 3,
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

        self.mds_results = {
            'mds_results': None,
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
        inst_btw_split_results = self.corr_btw_split(data, 4)

        subj_to_group_btw_split_results = self.corr_to_group_btw_split(data, 3)
        inst_to_group_btw_split_results = self.corr_to_group_btw_split(data, 4)

        if len(self.map_var) > 1:
            subj_btw_variable_results = self.corr_subj_btw_variable(data)
            subj_to_group_variable_results = self.corr_subj_to_group_variable(data)
        else:
            subj_btw_variable_results = None
            subj_to_group_variable_results = None

        # package results
        results = {
            "subj_btw_dataset" : subj_btw_dataset_results,
            "inst_btw_dataset" : inst_btw_dataset_results,
            "subj_to_group_dataset" : subj_to_group_dataset_results,
            "inst_to_group_dataset" : inst_to_group_dataset_results,
            "subj_btw_variable" : subj_btw_variable_results,
            "subj_to_group_btw_variable" : subj_to_group_variable_results,
        }
        return results


    def load_all(self):
        loaded = np.load(self.output_path / 'CorrMap_results.npz', allow_pickle=True)
        self.corr_maps = loaded['corr_maps'].item()
        self.corr_results = loaded['corr_results'].item()
        self.mds_results = loaded['mds_results'].item()


    def save_all(self):
        output = {
            'corr_maps': self.corr_maps,
            'corr_results': self.corr_results,
            'mds_results': self.mds_results,
        }
        output_path = self.output_path / 'CorrMap_results.npz'
        np.savez(output_path, **output)



    @staticmethod
    def _parallel_compute_corr_for_split(data, reps, metric, axis):
        """
        Helper function to compute correlation between split
        """
        metric_data = data[reps, :, metric]
        results = np.empty(shape=(data.shape[axis]))

        data_1 = metric_data[0,:,:]
        data_2 = metric_data[1,:,:]

        for ax in range(data.shape[axis]):
            if axis == 3:
                results[ax] = pearsonr(data_1[ax, :], data_2[ax, :]).statistic
            elif axis == 4:
                results[ax] = pearsonr(data_1[:, ax], data_2[:, ax]).statistic
        return reps, metric, results


    @staticmethod
    def _parallel_compute_corr_to_group_for_split(data, reps, test_subj, split1, split2, metric, axis):
        """
        Helper function to compute to-group correlation between split
        """
        subj_data = data[reps, split1, metric]
        all_but_subj_data = np.delete(data, test_subj, axis=axis)

        ### ZZZ Continue from here


    def corr_btw_split(self, data, axis):
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
        with ProcessPoolExecutor() as executor:
            futures = []
            for reps in range(data.shape[0]):
                for metric in range(data.shape[2]):
                    futures.append(executor.submit(self._parallel_compute_corr_for_split,
                        data, reps, metric, axis))
            for future in futures:
                reps, metric, result = future.result()
                results[reps, metric, :] = result
        return results


    def corr_to_group_btw_split(self, data, axis):
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
                                  data.shape[1],    # split half
                                  data.shape[2],    # metrics
                                  data.shape[axis], # subj/inst
                                  ))

        with ProcessPoolExecutor() as executor:
            futures = []
            for reps in range(data.shape[0]):
                for test_subj in range(data.shape[axis]):
                    for split1 in range(data.shape[1]):
                        for split2 in range(data.shape[1]):
                            for metric in range(data.shape[2]):
                                futures.append(executor.submit(self._parallel_compute_corr_to_group_for_split,
                                    data, reps, test_subj, split1, split2, metric, axis))

        ### ZZZ Continue from here


    @staticmethod
    def corr_subj_to_group_dataset(data):
        # data axis order: [human dataset 1/2, metrics, subj, model, values]
        # subject-to-group analysis, compare subj 1 to 2,3,4, ... then average
        # there are four possible ways to compute the group, they were computed separately and average at the end
        result = np.empty(shape = (data.shape[0], data.shape[0], data.shape[1], data.shape[2]))
        for test_subj in range(data.shape[2]):
            all_but_subj_data = np.delete(data, test_subj, axis=2)
            for dataset in range(data.shape[0]):
                for dataset2 in range(data.shape[0]):
                    for metric in range(data.shape[1]):
                        all_but_subj_results = np.empty(shape = (all_but_subj_data.shape[2]))
                        for other_subj in range(all_but_subj_data.shape[2]):
                            subj_data = data[dataset][metric][test_subj]
                            other_data = all_but_subj_data[dataset2][metric][other_subj]
                            all_but_subj_results[other_subj] = pearsonr(subj_data, other_data).statistic
                        all_but_subj_results = stat_func.r2z(all_but_subj_results, 'pearson')
                        all_but_subj_results = np.mean(all_but_subj_results, axis = 0)
                        result[dataset][dataset2][metric][test_subj] = all_but_subj_results
        result = np.mean(result, axis = 0)
        result = np.mean(result, axis = 0)
        result = stat_func.z2r(result, 'pearson')
        return result


    @staticmethod
    def corr_inst_to_group_dataset(data):
        # data axis order: [human dataset 1/2, metrics, subj, model, values]
        # inst-to-group analysis, compare inst 1 to 2, 3, 4, ... then average
        # there are four possible ways to compute the group, they were computed separately and average at the end
        result = np.empty(shape = (data.shape[0], data.shape[0], data.shape[1], data.shape[3]))
        for test_inst in range(data.shape[3]):
            all_but_inst_data = np.delete(data, test_inst, axis=3)
            for dataset in range(data.shape[0]):
                for dataset2 in range(data.shape[0]):
                    for metric in range(data.shape[1]):
                        all_but_inst_results = np.empty(shape = (all_but_inst_data.shape[3]))
                        for other_inst in range(all_but_inst_data.shape[3]):
                            inst_data = data[dataset][metric][:][test_inst]
                            other_data = all_but_inst_data[dataset2, metric, :, other_inst]
                            all_but_inst_results[other_inst] = pearsonr(inst_data, other_data).statistic
                        all_but_inst_results = stat_func.r2z(all_but_inst_results, 'pearson')
                        all_but_inst_results = np.mean(all_but_inst_results, axis = 0)
                        result[dataset][dataset2][metric][test_inst] = all_but_inst_results
        result = np.mean(result, axis = 0)
        result = np.mean(result, axis = 0)
        result = stat_func.z2r(result, 'pearson')
        return result


    @staticmethod
    def corr_subj_btw_variable(data):
        # data axis order: [human dataset 1/2, metrics, subj, model, values]
        # correlation between metrics for each subjects, without average dataset
        n_metrics = data.shape[1]
        metric_pairs = list(combinations([x for x in range(n_metrics)], 2))

        result = np.empty(shape=(data.shape[0], data.shape[0], len(metric_pairs), data.shape[2]))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                dataset1 = data[i]
                dataset2 = data[j]
                for subj in range(data.shape[2]):
                    subj_data_1 = dataset1[:, subj]
                    subj_data_2 = dataset2[:, subj]
                    for k, pair in enumerate(metric_pairs):
                        a, b = pair
                        result[i][j][k][subj] = pearsonr(subj_data_1[a], subj_data_2[b]).statistic
        return result


    @staticmethod
    def corr_subj_to_group_variable(data):
        # data axis order: [human dataset 1/2, metrics, subj, model, values]
        # subject-to-group analysis on metrics
        n_metrics = data.shape[1]
        metric_pairs = list(combinations([x for x in range(n_metrics)], 2))

        result = np.empty(shape=(data.shape[0], data.shape[0], len(metric_pairs), data.shape[2]))
        for test_subj in range(data.shape[2]):
            all_but_subj_data = np.delete(data, test_subj, axis=2)
            for dataset in range(data.shape[0]):
                for dataset2 in range(data.shape[0]):
                    all_but_subj_results = np.empty(shape = (len(metric_pairs), all_but_subj_data.shape[2]))
                    for other_subj in range(all_but_subj_data.shape[2]):
                        subj_data = data[dataset, :, test_subj]
                        other_data = all_but_subj_data[dataset2, :, other_subj]
                        for k, pair in enumerate(metric_pairs):
                            a, b = pair
                            all_but_subj_results[k][other_subj] = pearsonr(subj_data[a], other_data[b]).statistic
                    all_but_subj_results = stat_func.r2z(all_but_subj_results, 'pearson')
                    all_but_subj_results = np.mean(all_but_subj_results, axis = 1)
                    result[dataset, dataset2, :, test_subj] = all_but_subj_results
        result = stat_func.z2r(result, 'pearson')
        return result

