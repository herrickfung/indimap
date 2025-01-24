"""
Top-level wrapper for all analyses.
"""

from pathlib import Path
import numpy as np
import pickle

from .Maps.corr_map import CorrMap
from .Maps.rank_map import RankMap
from .Maps.top_map import TopMap


class IndiMap:
    def __init__(self, config):
        """
        Initialize analysis with a configuration dictionary.

        Parameters:
        --------------------------------------------------------------------------
        subj_data : pandas.DataFrame
            Human data (Pandas DataFrame).
        inst_data : pandas.DataFrame
            Model/Instance data (Pandas DataFrame).
        subj_column_name : str, optional
            Subject identifier in the DataFrame (default: 'subj').
        inst_column_name : str, optional
            Model/Instance identifier in the DataFrame (default: 'inst').
        map_variables : list of str, optional
            Column names to map/correlate on (default: ['acc', 'conf']).
        map_together : list of str, optional
            Variables to map/correlate together (e.g., image_index, stimulus).
        map_separate : list of str, optional
            Variables to map/correlate separately. The resulting map will be averaged after mapping.
        map_options : dict, optional
            Define which mapping analyses to perform (default: All True).
        bootstrap_iterations : int, optional
            Number of bootstrap iterations (default: 1000).
        bootstrap_seed : int, optional
            Seed for reproducibility (default: 42).
        load_exists : bool, optional
            True if existing results should be loaded (default: False).
        output_path : str, optional
            Path for storing output (default: 'results/').
        """

        default_config = {
            'subj_column_name': 'subj',
            'inst_column_name': 'inst',
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
        self.map_options = self.config['map_options']
        self.load_exists = self.config['load_exists']
        self.output_path = Path(self.config['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)

        """ Initialize all maps """
        self.corr_map = CorrMap(config)
        self.rank_map = RankMap(config)
        self.top_map = TopMap(config)

    def correlational_mapping(self):
        """ CorrMap analysis """
        self.corr_map.compute_corr_maps()
        self.corr_map.compute_corr_analysis()
        self.corr_map.save_all()

    def rank_based_mapping(self):
        """ RankMap analysis """
        corr_path = self.output_path / 'CorrMap_results.npz'
        if corr_path.is_file():
            self.rank_map.load_map_from_corr(self.output_path)
        else:
            self.rank_map.compute_corr_map()
        self.rank_map.compute_rank_analysis()
        self.rank_map.save_all(self.output_path)

    def top_based_mapping(self):
        """ TopMap analysis """
        corr_path = self.output_path / 'CorrMap_results.npz'
        if corr_path.is_file():
            self.top_map.load_map_from_corr(self.output_path)
        else:
            self.top_map.compute_corr_map()
        self.top_map.compute_top_analysis()
        self.top_map.save_all(self.output_path)

    def map(self):
        """Run/Load all results"""

        if self.map_options['CorrMap']:
            if self.load_exists:
                self.corr_map.load_all()
            else:
                self.correlational_mapping()

        if self.map_options['RankMap']:
            if self.load_exists:
                self.rank_map.load_all(self.output_path)
            else:
                self.rank_based_mapping()

        if self.map_options['TopMap']:
            if self.load_exists:
                self.top_map.load_all(self.output_path)
            else:
                self.top_based_mapping()
