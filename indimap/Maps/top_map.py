from scipy.stats import pearsonr
from itertools import combinations
from pathlib import Path
import numpy as np

from .util.stat_func import stat_func
from .corr_map import CorrMap

class TopMap:
    def __init__(self, config):
        self.config = config
        self.corr_subj_to_inst = None
        self.corr_subj_to_subj = None
        self.corr_inst_to_inst = None
        self.corr_adjusted_map = None

        self.top_maps = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
            'adjusted_map': None,
        }

        self.top_counts = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
            'adjusted_map': None,
        }

        self.top_correlations = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
            'adjusted_map': None,
        }

        self.top_results = {
            'subj_to_inst': None,
            'subj_to_subj': None,
            'inst_to_inst': None,
            'adjusted_map': None,
        }



    def save_all(self, path):
        output = {
            'top_maps': self.top_maps,
            'top_counts': self.top_counts,
            'top_correlations': self.top_correlations,
            'top_results': self.top_results,
        }
        output_path = path / 'TopMap_results.npz'
        np.savez(output_path, **output)


    def load_all(self, path):
        loaded = np.load(path / 'TopMap_results.npz', allow_pickle=True)
        self.top_maps = loaded['top_maps'].item()
        self.top_counts = loaded['top_counts'].item()
        self.top_correlations = loaded['top_correlations'].item()
        self.top_results = loaded['top_results'].item()


    def load_map_from_corr(self, path):
        corr_map = CorrMap(self.config)
        corr_map.load_all()

        self.corr_subj_to_inst = corr_map.corr_maps['subj_to_inst']
        self.corr_subj_to_subj = corr_map.corr_maps['subj_to_subj']
        self.corr_inst_to_inst = corr_map.corr_maps['inst_to_inst']
        self.corr_adjusted_map = corr_map.corr_maps['adjusted_map']


    def compute_corr_map(self):
        corr_map = CorrMap(self.config)
        corr_map.compute_corr_maps()

        self.corr_subj_to_inst = corr_map.corr_maps['subj_to_inst']
        self.corr_subj_to_subj = corr_map.corr_maps['subj_to_subj']
        self.corr_inst_to_inst = corr_map.corr_maps['inst_to_inst']
        self.corr_adjusted_map = corr_map.corr_maps['adjusted_map']


    def compute_top_map(self):
        # return nan except for retain
        if self.corr_subj_to_inst is not None:
            self.top_maps['subj_to_inst'] = self.shrink_mat(self.corr_subj_to_inst, same_xy = False)
        if self.corr_subj_to_subj is not None:
            self.top_maps['subj_to_subj'] = self.shrink_mat(self.corr_subj_to_subj, same_xy = True)
        if self.corr_inst_to_inst is not None:
            self.top_maps['inst_to_inst'] = self.shrink_mat(self.corr_inst_to_inst, same_xy = True)
        self.top_maps['adjusted_map'] = self.shrink_mat(self.corr_adjusted_map, same_xy = True)


    def compute_top_counts_corr(self):
        # get count per instance, and best correlation per subject
        name_dicts = ['subj_to_inst', 'subj_to_subj', 'inst_to_inst', 'adjusted_map']

        for name in name_dicts:
            if self.top_maps[name] is not None:
                self.top_counts[name], self.top_correlations[name] = self.get_counts_and_corr(self.top_maps[name])


    def compute_top_analysis(self):
        name_dicts = ['subj_to_inst', 'subj_to_subj', 'inst_to_inst', 'adjusted_map']
        for name in name_dicts:
            if name == 'adjusted_map':
                self.top_results[name] = self.do_top_analysis(name, True)
            else:
                self.top_results[name] = self.do_top_analysis(name)


    def do_top_analysis(self, key, is_adjust=False):
        count_corr_btw_dataset_results = self.corr_btw_dataset(self.top_counts[key])
        best_corr_btw_dataset_results = self.corr_btw_dataset(self.top_correlations[key], True, is_adjust)
        count_corr_btw_variable_results = self.corr_btw_variable(self.top_counts[key])
        best_corr_btw_variable_results = self.corr_btw_variable(self.top_correlations[key], True, is_adjust)

        results = {
            "count_corr_btw_dataset": count_corr_btw_dataset_results,
            "best_corr_btw_dataset": best_corr_btw_dataset_results,
            "count_corr_btw_variable": count_corr_btw_variable_results,
            "best_corr_btw_variable": best_corr_btw_variable_results,
        }
        return results



    @staticmethod
    def shrink_mat(data, same_xy):
        result = np.empty(shape=(data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
        for reps in range(data.shape[0]):
            for metric in range(data.shape[1]):
                sub_data = data[reps, metric, :, :]
                result[reps, metric, ...] = TopMap.retain_max_per_row(sub_data, retain=1, same_xy=same_xy)
        return result


    @staticmethod
    def retain_max_per_row(data, retain = 5, same_xy = True):
        retain = retain + 1
        # Copy the data to avoid modifying the original array
        max_only = np.full_like(data, np.nan)
        # Iterate over each row
        for row_idx in range(data.shape[0]):
            # Find the index of the maximum value in the row
            for i in range(1, retain):
                if same_xy == True:
                    max_col_idx = np.argsort(data[row_idx, :])[-i-1]
                else:
                    max_col_idx = np.argsort(data[row_idx, :])[-i]

                max_only[row_idx, max_col_idx] = data[row_idx, max_col_idx]
        return max_only


    @staticmethod
    def get_counts_and_corr(data):
        # data axis order: [human dataset 1/2, metrics, subj, model, values]
        count_results = np.empty(shape=(data.shape[0], data.shape[1], data.shape[3]))
        corr_results = np.empty(shape=(data.shape[0], data.shape[1], data.shape[2]))
        for reps in range(data.shape[0]):
            for metric in range(data.shape[1]):
                sub_data = data[reps, metric, ...]
                non_nan = ~np.isnan(sub_data)

                # get count
                count = np.sum(non_nan, axis = 0)
                count_results[reps, metric] = count

                # get best correlation value
                corr_results[reps, metric] = sub_data[non_nan]

        return count_results, corr_results


    @staticmethod
    def corr_btw_dataset(data, z_transform=False, is_adjust=False):
        if z_transform:
            if not is_adjust:
                data = stat_func.r2z(data, 'pearson')

        results = np.empty(shape = data.shape[1])
        for metric in range(data.shape[1]):
            result = pearsonr(data[0, metric], data[1, metric]).statistic
            results[metric] = result
        return results


    @staticmethod
    def corr_btw_variable(data, z_transform=False, is_adjust=False):
        if z_transform:
            if not is_adjust:
                data = stat_func.r2z(data, 'pearson')

        metric_pairs = list(combinations([x for x in range(data.shape[1])], 2))

        results = np.empty(shape = (data.shape[0], data.shape[0], len(metric_pairs)))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                set1 = data[i]
                set2 = data[j]
                for k, pair in enumerate(metric_pairs):
                    a, b = pair
                    results[i,j,k] = pearsonr(set1[a], set2[b]).statistic
        results = np.mean(results, axis = (0,1))
        return results

