from pathlib import Path
from math import comb
import numpy as np

from .corr_map import CorrMap

class RankMap:
    def __init__(self, config):

        default_config = {
            'map_metric': 'pearson',
            'map_options': {
                'CorrMap': True,
                'RankMap': True,
                'TopMap': True,
            },
            'CorrMap_options': {
                'subj_to_inst': True,
                'subj_to_subj': True,
                'subj_to_inst': False,
            },
            'rank_map_repeat_groupings': 1,
            'load_exists': False,
            'output_path': 'IndiMap_Result',
        }

        self.config = {**default_config, **config}

        self.human = self.config.get('subj_data')
        self.model = self.config.get('inst_data')
        self.metric = self.config.get('map_metric', 'pearson')

        self.human_iden = self.config.get('subj_column_name', 'subj')
        self.model_iden = self.config.get('inst_column_name', 'inst')
        self.repeat = self.config.get('human_repeat', 'reps')
        self.inst_repeat = self.config.get('inst_repeat')
        self.map_var = self.config.get('map_variables', ['acc', 'conf'])
        self.map_tgt = self.config.get('map_together')
        self.map_sep = self.config.get('map_separate', [None])
        self.output_path = Path(self.config['output_path'])

        self.corr_map_option = self.config['CorrMap_options']
        self.rank_map_repeat_groupings = config.get('rank_map_repeat_groupings')

        # results
        self.rank_results = None


    def compute_rank_analysis(self):
        map_types = ['subj_to_inst', 'subj_to_subj']
        if self.inst_repeat:
            map_types.append('inst_to_inst')

        results = {
            map_type: self.do_rank_analysis(map_type) for map_type in map_types
            }

        self.rank_results = results


    def do_rank_analysis(self, map_type):
        results = np.empty(shape = (self.rank_map_repeat_groupings, len(self.map_var)))

        humans = self.human[self.human_iden].unique()
        humans_half = len(humans) // 2
        models = self.model[self.model_iden].unique()
        models_half = len(models) // 2

        for grouping in range(self.rank_map_repeat_groupings):
            np.random.seed(grouping)
            np.random.shuffle(humans)
            np.random.shuffle(models)

            sampled_data = self._get_sampled_data(map_type, humans, humans_half, models, models_half)
            result = CorrMap.cross_map_matrix(*sampled_data, self.repeat, self.metric,
                                              self.map_var, self.map_tgt,
                                              self.map_sep
                                              )
            result = self.sum_of_ranked_corr_diff(result)
            results[grouping] = result
        return results


    def _get_sampled_data(self, map_type, humans, humans_half, models, models_half):
        if map_type == 'subj_to_inst':
            sampled_human = self.human[self.human[self.human_iden].isin(humans[:humans_half])]
            sampled_model = self.model[self.model[self.model_iden].isin(models[:models_half])]
            return sampled_human, sampled_model, self.human_iden, self.model_iden

        elif map_type == 'subj_to_subj':
            sampled_human1 = self.human[self.human[self.human_iden].isin(humans[:humans_half])]
            sampled_human2 = self.human[~self.human[self.human_iden].isin(humans[:humans_half])]
            return sampled_human1, sampled_human2, self.human_iden, self.human_iden

        elif map_type == 'inst_to_inst':
            sampled_model1 = self.model[self.model[self.model_iden].isin(models[:models_half])]
            sampled_model2 = self.model[~self.model[self.model_iden].isin(models[:models_half])]
            return sampled_model1, sampled_model2, self.model_iden, self.model_iden


    def load_all(self):
        loaded = np.load(self.output_path / 'RankMap_results.npz', allow_pickle=True)
        self.rank_results = loaded['rank_results'].item()


    def save_all(self):
        output = {
            'rank_results': self.rank_results,
        }
        output_path = self.output_path / 'RankMap_results.npz'
        np.savez(output_path, **output)


    @staticmethod
    def sum_of_ranked_corr_diff(data):
        # set up
        n_reps = data.shape[0]
        n_metrics = data.shape[1]
        n_pairs = comb(data.shape[-1], 2)
        data_diff = np.empty(shape=(n_reps, n_metrics, n_pairs, data.shape[-1]))

        # rank matrix, this tells how well each subj1 is fit with subj2
        # negate for descending order, add one for 1-based ranking
        # 1 is best
        data = np.argsort(-data, axis = 3).argsort(axis=3) + 1

        # get index for all pairs
        p1, p2 = np.triu_indices(data.shape[-1], k = 1)

        # compute the difference
        data_diff = data[:, :, p2] - data[:, :, p1]

        # multiply dataset 1 to dataset 2, dataset is axis 0
        result = np.prod(data_diff, axis = 0)
        # sum everything, except for the metrics axis
        # sum is normalized by total number of elements summed
        result = np.sum(result, axis=(1,2)) / np.prod(result.shape[1:])

        # the result is a single number for each metric
        # the higher the better, the more consistent is the mapping across dataset
        return result

