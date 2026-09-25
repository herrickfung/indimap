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
            stim_column_name: str, optional
                Stimulus identifier in the DataFrame (default: 'stim').
            resp_column_name: str, optional
                Response identifier in the DataFrame, used only for the confusion matrix (default: 'resp').
            map_variables : list of str, optional
                Column names to map/correlate on (default: ['acc', 'conf']).
                The first variable is treated as accuracy when checking for extreme performers.
            map_together : str
                Column name of the items to map/correlate together (e.g., image_index, stimulus).
            map_separate : str
                Column name of the conditions to map/correlate separately. The resulting map will be averaged after mapping.
            map_confusion: bool, optional
                Whether to compute and compare confusion matrix (default: False).
                Requires `stim_column_name` and `resp_column_name` columns in both DataFrames.
            map_category: bool, optional
                Whether or not to compute and compare across category mapping (default: False).
                Categories are read from `stim_column_name`.
            bootstrap_iterations : int, optional
                Number of bootstrap iterations (default: 1000).
            bootstrap_seed : int, optional
                Seed for reproducibility (default: 42).
            nComp_PCA : int, optional
                Number of components for PCA (default: 10).
            output_path : str, optional
                Path for storing output (default: 'IndiMap_Result').
            graph_path: str, optional
                Path for storing plots (default: 'IndiMap_Plots').
        """

        default_config = {
            'task_name': 'Task',
            'model_name': 'Model',
            'subj_column_name': 'subj',
            'inst_column_name': 'inst',
            'stim_column_name': 'stim',
            'resp_column_name': 'resp',
            'map_variables': ['acc', 'conf'],
            'map_confusion': False,
            'map_category': False,
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
        self.stim_iden = self.config.get('stim_column_name')
        self.map_var = self.config.get('map_variables')
        self.map_tgt = self.config.get('map_together')
        self.map_sep = self.config.get('map_separate')
        self.map_confusion = self.config.get('map_confusion')
        self.map_category = self.config.get('map_category')
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
Map confusion matrix:           {self.map_confusion}
Map category:                   {self.map_category}
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
            if not self.corr_map.check_exist():
                self.compute_corr()
            self.rank_map.load_map_from_corr(self.output_path)
            self.rank_map.compute_rank_analysis()
            self.rank_map.save_all(self.output_path)

    def compute_top(self, load_exists: bool = False): 
        """ TopMap analysis """
        if load_exists:
            self.top_map.load_all(self.output_path)
        else:
            if not self.corr_map.check_exist():
                self.compute_corr()
            self.top_map.load_map_from_corr(self.output_path)
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

    def get_corr_map(self, map_from: str, map_to: str, split_by: str = 'rand'):
        """
        Retrieve the correlation map for a specified pair of matrices.
        This function fetches the correlation map between two specified 
        matrices ("subj" or "inst") from the stored correlation maps.

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".
            split_by (str): The method for splitting the data, either "rand" for random image splits
                or "cate" for category-based splits (default: "rand"). "cate" requires
                `map_category=True` and is not available for "inst" to "inst".

        Returns:
        CorrelationMapping
            A named tuple containing:
            - dims (str): A description of the dimensions of the correlation map.
            - mat (numpy.ndarray or None): The correlation map matrix if it exists, 
              otherwise None.
        """

        CorrelationMapping = namedtuple("CorrelationMapping", ["dims", "mat"])

        maps = {
            'rand': self.corr_map.corr_maps,
            'cate': self.corr_map.cat_corr_maps,
        }
        split_dim = "bootstrap iterations" if split_by == 'rand' else "category splits"

        return CorrelationMapping(
            dims=(
            f"Dimensions: {split_dim} x split-half x metrics x "
            f"{map_from} x {map_to}"
            ),
            mat=maps[split_by].get(f"{map_from}_to_{map_to}", None)
        )

    def get_corr_results(self, map_from: str, map_to: str, target: str, btw: str, split_by: str = 'rand'):
        """
        Retrieve correlation results for a specific mapping and target.

        Args:
            map_from (str): The source mapping identifier, either "subj", or "inst".
            map_to (str): The target mapping identifier, either "subj", or "inst".
            target (str): The target variable for which correlation results are retrieved, 
                either "subj", "inst", "subj_gp", or "inst_gp". "subj" refers to the rows (map_from) 
                and "inst" refers to the columns (map_to) of the correlation map. The "_gp" variants
                compare each target against all other targets instead of itself.
            btw (str): The between-group comparison identifier, either "split" or "var".
            split_by (str): The method for splitting the data, either "rand" for random image splits
                or "cate" for category-based splits (default: "rand").

        Returns:
            namedtuple: A `CorrelationResults` namedtuple containing:
                - dims (str): A description of the dimensions of the correlation results.
                - mat (np.ndarray or None): The correlation results matrix for the specified parameters.
        """

        CorrelationResults = namedtuple("CorrelationResults", ["dims", "mat"])

        maps = {
            'rand': self.corr_map.corr_results,
            'cate': self.corr_map.cat_corr_results,
        }
        split_dim = "bootstrap iterations" if split_by == 'rand' else "category splits"
        met_dim = "metrics" if btw == 'split' else "metric pairs"
        tgt_dim = map_from if target.startswith('subj') else map_to

        return CorrelationResults(
            dims=(
            f"Dimensions: {split_dim} x {met_dim} x {tgt_dim}"
            ),
            mat=(maps[split_by].get(f"{map_from}_to_{map_to}") or {}
                 ).get(f"{target}_btw_{btw}", None)
        )

    def get_rank_results(self, map_from: str, map_to: str, btw: str, split_by: str = 'rand'):
        """
        Retrieve rank results from the rank map.

        Args:
            map_from (str): The source mapping identifier, either "subj", or "inst".
            map_to (str): The target mapping identifier, either "subj", or "inst".
            btw (str): The between-group comparison identifier, either "split" or "var".
            split_by (str): The method for splitting the data, either "rand" for random image splits
                or "cate" for category-based splits (default: "rand").

        Returns:
            RankResults: A named tuple containing:
                - dims (str): Description of the dimensions of the rank results.
                - mat (np.ndarray or None): The rank results matrix corresponding to the specified 
                  mapping and between-group comparison.
        """

        split_by_prefix = "cat_" if split_by == "cate" else ""
        split_dim = "bootstrap iterations" if split_by == 'rand' else "category splits"
        met_dim = "metrics" if btw == 'split' else "metric pairs"
        RankResults = namedtuple("RankResults", ["dims", "mat"])
        return RankResults(
            dims=(
            f"Dimensions: {split_dim} x {met_dim}"
            ),
            mat=((self.rank_map.rank_results or {}).get(f"{map_from}_to_{map_to}") or {}
                 ).get(f"{split_by_prefix}btw_{btw}", None)
        )

    def get_top_map(self, map_from: str, map_to: str):
        """
        Retrieve the correlation map for a specified pair of matrices,
        retaining only the best mapped target for each source matrix. 

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".

        Returns:
        TopMapping
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
        Retrieve the number of times each target is the best mapped target of a source.

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".

        Returns:
        TopCount
            A named tuple containing:
            - dims (str): A description of the dimensions of the count matrix.
            - mat (numpy.ndarray or None): The count matrix if it exists, 
              otherwise None.
        """

        TopCount = namedtuple("TopCount", ["dims", "mat"])
        return TopCount(
            dims=(
            "Dimensions: bootstrap iterations x split-half x metrics x "
            f"{map_to}"
            ),
            mat=self.top_map.top_ct.get(f"{map_from}_to_{map_to}", None)
        )

    def get_top_corr(self, map_from: str, map_to: str):
        """
        Retrieve the correlation values of the best mapped target for each source matrix.

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".

        Returns:
        TopCorr
            A named tuple containing:
            - dims (str): A description of the dimensions of the correlation values.
            - mat (numpy.ndarray or None): The correlation values if they exist, 
              otherwise None.
        """

        TopCorr = namedtuple("TopCorr", ["dims", "mat"])
        return TopCorr(
            dims=(
            "Dimensions: bootstrap iterations x split-half x metrics x "
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
        TopResults
            A named tuple containing:
            - dims (str): A description of the dimensions of the result matrix.
            - mat (numpy.ndarray or None): The result matrix if it exists, 
              otherwise None.
        """

        met_dim = "metrics" if btw == 'split' else "metric pairs"
        TopResults = namedtuple("TopResults", ["dims", "mat"])
        return TopResults(
            dims=(
            f"Dimensions: bootstrap iterations x {met_dim}"
            ),
            mat=(self.top_map.top_results.get(f"{map_from}_to_{map_to}") or {}
                 ).get(f"{corr_on}_btw_{btw}", None)
        )

    def get_top_expo(self, map_from: str, map_to: str):
        """
        Retrieve the top map exponential count distribution results

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".

        Returns:
        TopExpo
            Intercept and slope of the exponential distribution fit for each metric.
            A named tuple containing:
            - dims (str): A description of the dimensions of the result matrix.
            - mat (numpy.ndarray or None): The result matrix if it exists, 
              otherwise None.
        """

        TopExpo = namedtuple("TopExpo", ["dims", "mat"])
        return TopExpo(
            dims=(
            "Dimensions: (bootstrap iterations x split-half) x metrics x intercept/slope"
            ),
            mat=(self.top_map.top_results.get(f"{map_from}_to_{map_to}") or {}
                 ).get('expo_slope', None)
        )


    def get_top_iden(self, map_from: str, map_to: str, target: str):
        """
        Retrieve the results for identifiability analyses.
        The best mapped target of each source is identified in the first split-half,
        and its correlation is evaluated in the second split-half.

        Args:
            map_from (str): The source matrix identifier, either "subj" or "inst".
            map_to (str): The target matrix identifier, either "subj" or "inst".
            target (str): Either "pair" for the correlation with the best mapped target,
                or "gp" for the mean correlation with all other targets.

        Returns:
        TopIden
            A named tuple containing:
            - dims (str): A description of the dimensions of the result matrix.
            - mat (numpy.ndarray or None): The result matrix if it exists, 
              otherwise None.
        """

        TopIden = namedtuple("TopIden", ["dims", "mat"])
        return TopIden(
            dims=(
            "Dimensions: bootstrap iterations x metrics x "
            f"{map_from}"
            ),
            mat=(self.top_map.top_results.get(f"{map_from}_to_{map_to}") or {}
                 ).get(f"top_{target}_btw_split", None)
        )

    def get_mds(self):
        """
        Retrieve the results of the Multi-Dimensional Scaling (MDS) analysis.
        Returns:
            namedtuple: An MDS_Results namedtuple containing:
                - dims (str): A description of the dimensions in the MDS results.
                - mat (array-like): The MDS results matrix. If the number of subjects and
                  instances differ, the smaller group is padded with nan.
        """

        MDS_Results = namedtuple("MDS_Results", ["dims", "mat"])
        return MDS_Results(
            dims=(
            "Dimensions: metrics x human/model x max(N_subjs, N_insts) x 2 MDS dimensions"
            ),
            mat=self.dims_map.mds_results
        )

    def get_pca_results(self, proj_to: str):
        """
        Retrieve split-half PCA results.
        PCA is fitted on a random half of the subjects, and the explained variance
        of each component is computed after projecting the specified data onto it.

        Args:
            proj_to (str): The data projected onto the PCA components, either
                "same_human" (the half of subjects used for fitting),
                "diff_human" (the held-out half of subjects),
                "scrm_human" (the fitted half of subjects with image order scrambled), or
                "model" (a random half of the instances).

        Returns:
            namedtuple: A named tuple `PCA_Results` containing:
                - dims (str): Description of the dimensions of the PCA results.
                - mat (numpy.ndarray or None): Explained variance of each PCA component.
        """

        if proj_to in ['same_human', 'diff_human']:
            dims = "Dimensions: conditions x metrics x PCA components"
        else:
            dims = "Dimensions: bootstrap iterations x conditions x metrics x PCA components"

        PCA_Results = namedtuple("PCA_Results", ["dims", "mat"])
        return PCA_Results(
            dims=dims,
            mat=(self.dims_map.split_half_pca_results or {}).get(f"proj_{proj_to}_var", None)
        )

    def get_pred_results(self, by: str, using: str, within_metric: bool):
        """
        Retrieve prediction results based on specified parameters.
        This function fetches prediction results from the `pred_map` attribute
        using the specified grouping, method, and metric type.
        Each human subject's behavior is predicted from other subjects or from instances.

        Args:
            by (str): Perform prediction by this variable, either 'subj' or 'inst'.
            using (str): The method used for prediction. Options include:
                         'rand', 'avg', 'corr'.
                1. 'rand' - Random individual.
                2. 'avg' - Average of all individuals.
                3. 'corr' - Weighted average of all individuals by CorrMap.
            within_metric (bool): If True, retrieves results for within metrics predictions.
                                  If False, retrieves results across metrics predictions.

        Returns:
            namedtuple: A named tuple `PredResults` containing:
                - dims (str): Description of the dimensions of the prediction results.
                - metric (list): The order of the metric dimension. For across metric predictions,
                  each pair is (predictor metric, predicted metric).
                - mat (np.ndarray or None): The prediction results matrix retrieved from `pred_map`.
        """

        if within_metric:
            met_type = 'within'
            metric_pair = self.map_var
            dims = 'Dimensions: bootstrap iterations x metrics x subj'
        else:
            met_type = 'across'
            metric_pair = list(permutations(self.map_var, 2))
            dims = 'Dimensions: bootstrap iterations x metric pairs x subj'

        PredResults = namedtuple("PredResults", ["dims", "metric", "mat"])
        return PredResults(
            dims=dims,
            metric=metric_pair,
            mat=self.pred_map.pred_results[met_type][by][using]
        )
