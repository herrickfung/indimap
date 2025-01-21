from pathlib import Path
import numpy as np
import pickle

from .Maps.corr_map import CorrMap
from .Maps.rank_map import RankMap
from .Maps.top_map import TopMap


class IndiMap:
    def __init__(self, config):
        '''
        initalize analysis with a configuration dictionary
        -----------------------------------------------------------------------
        Parameters:
        -----------------------------------------------------------------------
        subj_data: Human data
        inst_data: Model/Instance data
        map_metric: Metric to perform initial mapping, default: 'pearson'
        subj_column_name: Subject Identifier in the Dataset, default: 'subj'
        inst_column_name: Model/Instance Identifier in the Dataset, default: 'inst'
        human_repeat: Repetition column name in the Dataset, default: 'reps'
        map_variables: Columns name to map/correlate on, default: ['acc', 'conf']
        map_together: Variables that will map/correlate together, e.g. image_index, stimulus
        map_separate: Variables that will map/correlate separately. The resulting map will be average after mapping.
        map_options: Define which mapping analyses were performed. Default: All True
        corr_map_option: Define options for correlation mapping.
        output_path: Path for storing output. default: 'results/'
        '''

        default_config = {
            'map_options': {
                'CorrMap': True,
                'RankMap': True,
                'TopMap': True,
            },
            'bootstrap_iterations': 1000,
            'bootstrap_seed': 42,
            'load_exists': False,
            'output_path': 'IndiMap_Result',
        }

        self.config = {**default_config, **config}

        self.subj = self.config['subj_data']
        self.inst = self.config['inst_data']
        self.subj_column_name = self.config['subj_column_name']
        self.inst_column_name = self.config['inst_column_name']
        self.map_variables = self.config['map_variables']
        self.map_together = self.config['map_together']
        self.map_separate = self.config['map_separate']

        self.map_options = self.config['map_options']
        self.load_exists = self.config['load_exists']
        self.output_path = Path(self.config['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize analysis
        self.corr_map = CorrMap(config)
        # self.rank_map = RankMap(config)
        # self.top_map = TopMap(config)


    def correlational_mapping(self):
        self.corr_map.compute_corr_maps()
        self.corr_map.compute_corr_analysis()
        # self.corr_map.compute_mds()
        # self.corr_map.save_all()


    def rank_based_mapping(self):
        self.rank_map.compute_rank_analysis()
        self.rank_map.save_all()


    def top_based_mapping(self):
        corr_path = self.output_path / 'CorrMap_results.npz'

        if corr_path.is_file():
            self.top_map.load_map_from_corr(self.output_path)
        else:
            self.top_map.compute_corr_map()

        self.top_map.compute_top_map()
        self.top_map.compute_top_counts_corr()
        self.top_map.compute_top_analysis()
        self.top_map.save_all(self.output_path)


    def map(self):
        '''run analysis'''

        if self.map_options['CorrMap']:
            if self.load_exists:
                self.corr_map.load_all()
            else:
                self.correlational_mapping()

        if self.map_options['RankMap']:
            if self.load_exists:
                self.rank_map.load_all()
            else:
                self.rank_based_mapping()

        if self.map_options['TopMap']:
            if self.load_exists:
                self.top_map.load_all(self.output_path)
            else:
                self.top_based_mapping()
