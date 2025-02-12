"""
Top-level wrapper for all analyses.
"""

from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle

from .Maps.corr_map import CorrMap
from .Maps.rank_map import RankMap
from .Maps.top_map import TopMap
from .Maps.dims_map import DimsMap

class IndiMap:
    def __init__(self, config):
        """
        Initialize analysis with a configuration dictionary.

        Parameters:
        --------------------------------------------------------------------------
        task_name: str, optional
            Task name (default: 'Task').
        model_name : str, optional
            Model name (default: 'Model').
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
        bootstrap_iterations : int, optional
            Number of bootstrap iterations (default: 1000).
        bootstrap_seed : int, optional
            Seed for reproducibility (default: 42).
        nComp_PCA : int, optional
            Number of components for PCA (default: 10).
        output_path : str, optional
            Path for storing output (default: 'IndiMap_Result/').
        graph_path: str, optional
            Path for storing plots (default: 'IndiMap_Plots/').
        """

        default_config = {
            'task_name': 'Task',
            'model_name': 'Model',
            'subj_column_name': 'subj',
            'inst_column_name': 'inst',
            'map_variables': ['acc', 'conf'],
            'bootstrap_iterations': 1000,
            'bootstrap_seed': 42,
            'nComp_PCA': 10,
            'output_path': 'IndiMap_Result',
            'graph_path': 'IndiMap_Plots',
        }

        self.config = {**default_config, **config}

        self.task_name = self.config.get('task_name')
        self.model_name = self.config.get('model_name')
        self.human = self.config.get('subj_data')
        self.model = self.config.get('inst_data')
        self.human_iden = self.config.get('subj_column_name')
        self.model_iden = self.config.get('inst_column_name')
        self.map_var = self.config.get('map_variables')
        self.map_tgt = self.config.get('map_together')
        self.map_sep = self.config.get('map_separate')
        self.n_bs = self.config['bootstrap_iterations']
        self.bs_seed = self.config['bootstrap_seed']
        self.n_comps = self.config['nComp_PCA']
        self.output_path = Path(self.config['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.graph_path = Path(self.config['graph_path'])
        self.graph_path.mkdir(parents=True, exist_ok=True)

        """ Initialize all maps """
        self.corr_map = CorrMap(self.config)
        self.rank_map = RankMap(self.config)
        self.top_map = TopMap(self.config)
        self.dims_map = DimsMap(self.config)

    def __str__(self):
        n_subjs = self.human[self.human_iden].nunique()
        n_insts = self.model[self.model_iden].nunique()
        n_imgs = self.human[self.map_tgt].nunique()
        n_conds = self.human[self.map_sep].nunique()
        return f"""
--------------------------------------------------------------------------------
Individual Differences Mapping (IndiMap) analyses
--------------------------------------------------------------------------------
Dataset Name:                   {self.model_name.capitalize()} on {self.task_name.capitalize()}
Number of subjects:             {n_subjs}
Number of instances:            {n_insts}
Number of Conditions:           {n_conds}
Number of Images:               {n_imgs}
Mapping variables:              {self.map_var}
Mapping together:               {self.map_tgt}
Mapping separately:             {self.map_sep}
Bootstrap iterations:           {self.n_bs}
Bootstrap random seed:          {self.bs_seed}
Number of components (PCA):     {self.n_comps}
--------------------------------------------------------------------------------
CorrMap Exist:                  {self.corr_map.check_exist()}
RankMap Exist:                  {self.rank_map.check_exist(self.output_path)}
TopMap Exist:                   {self.top_map.check_exist(self.output_path)}
DimsMap Exist:                  {self.dims_map.check_exist()}
--------------------------------------------------------------------------------
Output path:                    {self.output_path}
Graph path:                     {self.graph_path}  
--------------------------------------------------------------------------------
"""

    def compute_corr(self, load_exists = False):
        """ CorrMap analysis """
        if load_exists:
            self.corr_map.load_all()
        else:
            self.corr_map.compute_corr_maps()
            self.corr_map.compute_corr_analysis()
            self.corr_map.save_all()

    def compute_rank(self, load_exists = False):
        """ RankMap analysis """
        if load_exists:
            self.rank_map.load_all(self.output_path)
        else:
            corr_path = self.output_path / 'CorrMap_results.npz'
            if corr_path.is_file():
                self.rank_map.load_map_from_corr(self.output_path)
            else:
                self.rank_map.compute_corr_map()
            self.rank_map.compute_rank_analysis()
            self.rank_map.save_all(self.output_path)

    def compute_top(self, load_exists = False):
        """ TopMap analysis """
        if load_exists:
            self.top_map.load_all(self.output_path)
        else:
            corr_path = self.output_path / 'CorrMap_results.npz'
            if corr_path.is_file():
                self.top_map.load_map_from_corr(self.output_path)
            else:
                self.top_map.compute_corr_map()
            self.top_map.compute_top_analysis()
            self.top_map.save_all(self.output_path)

    def compute_dims(self, load_exists = False):
        """ DimsMap analysis """
        if load_exists:
            self.dims_map.load_all(self.output_path)
        else:
            self.dims_map.compute_dims_analysis()
            self.dims_map.save_all(self.output_path)

    def compute_all(self, load_exists = False):
        """Compute all results"""
        tasks = [
            ('CorrMap', self.compute_corr),
            ('RankMap', self.compute_rank),
            ('TopMap', self.compute_top),
            ('DimsMap', self.compute_dims),
        ]

        if load_exists:
            print("Loading existing results ...")
            for _, func in tasks:
                func(load_exists)
        else:
            for name, func in tqdm(tasks):
                print(f"Computing {name}")
                func(load_exists)

    def plot_all(self):
        """TDL: Match it with compute_all"""
        """ Plot all results """
        tasks = [
            ('CorrMap', self.corr_map.plot_all),
            ('RankMap', self.rank_map.plot_all),
            ('TopMap', self.top_map.plot_all),
            ('DimsMap', self.dims_map.plot_all),
        ]
        for name, func in tasks:
            print(f"Plotting {name}")
            func()
