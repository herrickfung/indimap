"""
Top-level wrapper for all analyses.
"""

from itertools import permutations
from tqdm import tqdm
from collections import namedtuple
from pathlib import Path

from indimap.maps import CorrMap, RankMap, TopMap, DimsMap, PredMap

class IndiMap:
    def __init__(self, config: dict):
        """
        Initialize analysis with a configuration dictionary.

        Parameters:
        --------------------------------------------------------------------------
        config: dict
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

        self.n_subjs = self.human[self.human_iden].nunique()
        self.n_insts = self.model[self.model_iden].nunique()
        self.n_metrics = len(self.map_var)
        self.n_imgs = self.human[self.map_tgt].nunique()
        self.n_conds = self.human[self.map_sep].nunique()

        """ Initialize all maps """
        self.corr_map = CorrMap(self.config)
        self.rank_map = RankMap(self.config)
        self.top_map = TopMap(self.config)
        self.dims_map = DimsMap(self.config)
        self.pred_map = PredMap(self.config)

    def __str__(self):
        return f"""
--------------------------------------------------------------------------------
Individual Differences Mapping (IndiMap) analyses
--------------------------------------------------------------------------------
Dataset Name:                   {self.model_name.capitalize()} on {self.task_name.capitalize()}
Number of subjects:             {self.n_subjs}
Number of instances:            {self.n_insts}
Number of Conditions:           {self.n_conds}
Number of Images:               {self.n_imgs}
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
PredMap Exist:                  {self.pred_map.check_exist()}
--------------------------------------------------------------------------------
Output path:                    {self.output_path}
Graph path:                     {self.graph_path}  
--------------------------------------------------------------------------------
"""

    def compute_corr(self, load_exists: bool = False):
        """ CorrMap analysis """
        if load_exists:
            self.corr_map.load_all()
        else:
            self.corr_map.compute_corr_maps()
            self.corr_map.compute_corr_analysis()
            self.corr_map.save_all()

    def compute_rank(self, load_exists: bool = False):
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

    def compute_top(self, load_exists: bool = False): 
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

    def compute_dims(self, load_exists: bool = False):
        """ DimsMap analysis """
        if load_exists:
            self.dims_map.load_all(self.output_path)
        else:
            self.dims_map.compute_dims_analysis()
            self.dims_map.save_all(self.output_path)

    def compute_pred(self, load_exists: bool = False):
        """ Prediction analyses """
        if load_exists:
            self.pred_map.load_all(self.output_path)
        else:
            self.pred_map.compute_pred_maps()
            self.pred_map.save_all(self.output_path)

    def compute_all(self, load_exists: bool = False):
        """Compute all results"""
        tasks = [
            ('CorrMap', self.compute_corr),
            ('RankMap', self.compute_rank),
            ('TopMap', self.compute_top),
            ('DimsMap', self.compute_dims),
            ('PredMap', self.compute_pred),
        ]

        if load_exists:
            print("Loading existing results ...")
            for _, func in tasks:
                func(load_exists)
        else:
            for name, func in tqdm(tasks):
                print(f"Computing {name}")
                func(load_exists)

    def plot_corr(self):
        """ Plot correlation map """
        self.corr_map.plot_map_average()
        self.corr_map.plot_btw_split()
        self.corr_map.plot_btw_var()

    def plot_rank(self):
        """ Plot rank map """
        self.rank_map.plot_btw_split()
        self.rank_map.plot_btw_var()

    def plot_top(self):
        """ Plot top map """
        self.top_map.plot_top_average()
        self.top_map.plot_btw_split() 
        self.top_map.plot_btw_var()

    def plot_dims(self):
        """ Plot dims map """
        self.dims_map.plot_mds()
        self.dims_map.plot_pca_explained_var()
    
    def plot_pred(self):
        """ Plot prediction results """
        self.pred_map.plot_wn_var()
        self.pred_map.plot_btw_var()

    def plot_all(self):
        """ Plot all results """
        tasks = [
            ('CorrMap', self.plot_corr),
            ('RankMap', self.plot_rank),
            ('TopMap', self.plot_top),
            ('DimsMap', self.plot_dims),
            ('PredMap', self.plot_pred),
        ]
        for name, func in tasks:
            print(f"Plotting {name}")
            func()

    def get_corr_map(self, map_from: str, map_to: str):
        """
        Retrieve the correlation map for a specified pair of matrices.
        This function fetches the correlation map between two specified 
        matrices ("subj" or "inst") from the stored correlation maps.

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".

        Returns:
        CorrelationMapping
            A named tuple containing:
            - dims (str): A description of the dimensions of the correlation map.
            - mat (numpy.ndarray or None): The correlation map matrix if it exists, 
              otherwise None.
        """

        CorrelationMapping = namedtuple("CorrelationMapping", ["dims", "mat"])
        return CorrelationMapping(
            dims=(
            "Dimensions: bootstrap iterations x split-half x metrics x "
            f"{map_from} x {map_to}"
            ),
            mat=self.corr_map.corr_maps.get(f"{map_from}_to_{map_to}", None)
        )

    def get_corr_results(self, map_from: str, map_to: str, target: str, btw: str):
        """
        Retrieve correlation results for a specific mapping and target.

        Args:
            map_from (str): The source mapping identifier, either "subj", or "inst".
            map_to (str): The target mapping identifier, either "subj", or "inst".
            target (str): The target variable for which correlation results are retrieved, either "subj", or "inst".
            btw (str): The between-group comparison identifier, either "split" or "var".

        Returns:
            namedtuple: A `CorrelationResults` namedtuple containing:
                - dims (str): A description of the dimensions of the correlation results.
                - mat (np.ndarray): The correlation results matrix for the specified parameters.
        """

        CorrelationResults = namedtuple("CorrelationResults", ["dims", "mat"])
        return CorrelationResults(
            dims=(
            f"Dimensions: bootstrap iterations x metrics x {target}"
            ),
            mat=self.corr_map.corr_results.get(
                f"{map_from}_to_{map_to}", {}
                ).get(f"{target}_btw_{btw}", None)
        )

    def get_rank_results(self, mat_from: str, map_to: str, btw: str):
        """
        Retrieve rank results from the rank map.

        Args:
            map_from (str): The source mapping identifier, either "subj", or "inst".
            map_to (str): The target mapping identifier, either "subj", or "inst".
            btw (str): The between-group comparison identifier, either "split" or "var".

        Returns:
            RankResults: A named tuple containing:
                - dims (str): Description of the dimensions of the rank results.
                - mat (np.ndarray): The rank results matrix corresponding to the specified 
                  mapping and between-group comparison.
        """

        RankResults = namedtuple("RankResults", ["dims", "mat"])
        return RankResults(
            dims=(
            "Dimensions: bootstrap iterations x metrics"
            ),
            mat=self.rank_map.rank_results.get(
                f"{mat_from}_to_{map_to}", {}
                ).get(f"btw_{btw}", None)
        )

    def get_top_map(self, map_from: str, map_to: str):
        """
        Retrieve the correlation map for a specified pair of matrices,
        retaining only the best mapped target for each source matrix. 

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".

        Returns:
        CorrelationMapping
            A named tuple containing:
            - dims (str): A description of the dimensions of the correlation map.
            - mat (numpy.ndarray or None): The correlation map matrix if it exists, 
              otherwise None.
        """

        TopMapping = namedtuple("TopMapping", ["dims", "mat"])
        return TopMapping(
            dims=(
            "Dimensions: bootstrap iterations x split-half x metrics x "
            f"{map_from} x {map_to}"
            ),
            mat=self.top_map.top_maps.get(f"{map_from}_to_{map_to}", None)
        )

    def get_top_ct(self, map_from: str, map_to: str):
        """
        Retrieve the count of the best mapped target for each source matrix.

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".

        Returns:
        CorrelationMapping
            A named tuple containing:
            - dims (str): A description of the dimensions of the correlation map.
            - mat (numpy.ndarray or None): The correlation map matrix if it exists, 
              otherwise None.
        """

        TopCount = namedtuple("TopCount", ["dims", "mat"])
        return TopCount(
            dims=(
            "Dimensions: bootstrap iterations x split-half x metrc"
            f"{map_from}"
            ),
            mat=self.top_map.top_ct.get(f"{map_from}_to_{map_to}", None)
        )

    def get_top_corr(self, map_from: str, map_to: str):
        """
        Retrieve the correlation valules of the best mapped target for each source matrix.

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".

        Returns:
        CorrelationMapping
            A named tuple containing:
            - dims (str): A description of the dimensions of the correlation map.
            - mat (numpy.ndarray or None): The correlation map matrix if it exists, 
              otherwise None.
        """

        TopCorr = namedtuple("TopCorr", ["dims", "mat"])
        return TopCorr(
            dims=(
            "Dimensions: bootstrap iterations x split-half x metrc x"
            f"{map_from}"
            ),
            mat=self.top_map.top_corr.get(f"{map_from}_to_{map_to}", None)
        )

    def get_top_results(self, map_from: str, map_to: str, btw: str, corr_on: str):
        """
        Retrieve the top map results.

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".
            btw (str): The between-group comparison identifier, either "split" or "var".
            corr_on (str): The metric to correlate on, either "ct", or "corr".

        Returns:
        CorrelationMapping
            A named tuple containing:
            - dims (str): A description of the dimensions of the correlation map.
            - mat (numpy.ndarray or None): The result matrix if it exists, 
              otherwise None.
        """

        TopResults = namedtuple("TopResults", ["dims", "mat"])
        return TopResults(
            dims=(
            "Dimensions: bootstrap iterations x metrics"
            ),
            mat=self.top_map.top_results.get(
                f"{map_from}_to_{map_to}", {}
                ).get(f"{corr_on}_btw_{btw}", None)
        )

    def get_top_expo(self, map_from: str, map_to: str):
        """
        Retrieve the top map exponential count distributuion results

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".

        Returns:
        Slope and intercept of the exponential distribution fit for each metric
            A named tuple containing:
            - dims (str): A description of the dimensions of the correlation map.
            - mat (numpy.ndarray or None): The result matrix if it exists, 
              otherwise None.
        """

        TopExpo = namedtuple("TopExpo", ["dims", "mat"])
        return TopExpo(
            dims=(
            "Dimensions: repetitions x metrics x intercept/slope"
            ),
            mat=self.top_map.top_results.get(f"{map_from}_to_{map_to}", {}
            ).get('expo_slope', None)
        )

    def get_mds(self):
        """
        Retrieve the results of the Multi-Dimensional Scaling (MDS) analysis.
        Returns:
            namedtuple: An MDS_Results namedtuple containing:
                - dims (str): A description of the dimensions in the MDS results.
                - mat (array-like): The MDS results matrix
        """

        MDS_Results = namedtuple("MDS_Results", ["dims", "mat"])
        return MDS_Results(
            dims=(
            "Dimensions: metrics x human/model x N_subjs x 2 MDS dimensions"
            ),
            mat=self.dims_map.mds_results
        )

    def get_pca_results(self, fit_on: str, proj_to: str, center: bool, scramble: bool):
        """
        Retrieve PCA results based on specified parameters.

        Args:
            fit_on (str): The dataset or condition on which the PCA was fitted (e.g., 'human', 'model').
            proj_to (str): The projection target (e.g., 'human', 'model').
            center (bool): If True, use centered PCA results; otherwise, use uncentered PCA results.
            scramble (bool): If True, use scrambled projection results; otherwise, use standard projection results.

        Returns:
            namedtuple: A named tuple `PCA_Results` containing:
                - dims (str): Description of the dimensions of the PCA results.
                - mat (numpy.ndarray): The PCA results matrix corresponding to the specified parameters.
        """

        if center:
            center_text = 'centered'
        else:
            center_text = 'uncentered'
        if scramble:
            proj = f'P_S_{proj_to}'
        else:
            proj = f'P_{proj_to}'

        PCA_Results = namedtuple("PCA_Results", ["dims", "mat"])
        return PCA_Results(
            dims=(
            "Dimensions: bootstrap iterations x conditions x metrics x \
            x PCA components x human/model"
            ""
            ),
            mat=self.dims_map.pca_results[center_text][fit_on][proj]
        )

    def get_pred_results(self, by: str, using: str, within_metric: bool):
        """
        Retrieve prediction results based on specified parameters.
        This function fetches prediction results from the `pred_map` attribute
        using the specified grouping, method, and metric type.

        Args:
            by (str): Perform prediction by this variable, either 'subj' or 'inst'.
            using (str): The method used for prediction. Options include:
                         'rand', 'avg', 'corr', 'ols', 'lasso', 'ridge'.
                1. 'rand' - Random individual.
                2. 'avg' - Average of all individuals.
                3. 'corr' - Weighted average of all individuals by CorrMap.
                4. 'ols' - Ordinary Least Squares regression.
                5. 'lasso' - L1 Lasso regression.
                6. 'ridge' - L2 Ridge regression.
            within_metric (bool): If True, retrieves results for within metrics predictions.
                                  If False, retrieves results across metrics predictions.

        Returns:
            namedtuple: A named tuple `PredResults` containing:
                - dims (str): Description of the dimensions of the prediction results.
                - metric (list or str): The order of the metric dimension.
                - mat (Any): The prediction results matrix retrieved from `pred_map`.
        """

        if within_metric:
            met_type = 'within'
            metric_pair = self.map_var
        else:
            met_type = 'across'
            metric_pair = list(permutations(self.map_var, 2))

        if using in ['rand', 'avg', 'corr']:
            dims = 'Dimensions: bootstrap iterations x metrics x subj'
        else:
            dims = 'Dimensions: condition x metrics x subj'

        PredResults = namedtuple("PredResults", ["dims", "metric", "mat"])
        return PredResults(
            dims=dims,
            metric=metric_pair,
            mat=self.pred_map.pred_results[met_type][by][using]
        )