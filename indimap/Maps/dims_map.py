'''
contains all functions related to dimension analyses
'''

from matplotlib import rcParams
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import einops
import numpy as np

from .util import map_func
rcParams['font.family'] = 'CMU Sans Serif'


class DimsMap:
    def __init__(self, config):
        self.config = config
        self.human = self.config.get('subj_data')
        self.model = self.config.get('inst_data')
        self.human_iden = self.config.get('subj_column_name')
        self.model_iden = self.config.get('inst_column_name')
        self.map_var = self.config.get('map_variables')
        self.map_tgt = self.config.get('map_together')
        self.map_sep = self.config.get('map_separate')
        self.n_comps = self.config.get('nComp_PCA')
        self.bs_seed = self.config.get('bootstrap_seed')
        self.output_path = Path(self.config['output_path'])
        self.graph_path = Path(self.config['graph_path'])

        # results
        self.pca_objects = None
        self.pca_results = None
        self.mds_results = None

        self.human_arr = None
        self.model_arr = None

    def check_exist(self):
        """ Check whether the file exist """
        file_path = self.output_path / 'DimsMap_results.npz'
        return file_path.exists()

    def load_all(self, path):
        """ Loads precomputed results from a file """
        loaded = np.load(path / 'DimsMap_results.npz', allow_pickle=True)
        self.pca_objects = loaded['PCA_objs']
        self.pca_results = loaded['PCA_results']
        self.mds_results = loaded['MDS']

    def save_all(self, path):
        """ Save results to a file """
        output = {
            'PCA_objs': self.pca_objects,
            'PCA_results': self.pca_results,
            'MDS': self.mds_results,
        }
        output_path = path / 'DimsMap_results.npz'
        np.savez(output_path, **output)

    def compute_dims_analysis(self):
        """Perform rank analyses on all maps."""
        self._convert_data_array()
        self._compute_pca()
        self._compute_mds()
        self.save_all(self.output_path)

    def _convert_data_array(self) -> None:
        # convert dataframe to array
        self.human_arr = map_func.convert_to_array(self.human, self.human_iden,
                                                   self.map_var, self.map_tgt,
                                                   self.map_sep
                                                   )
        self.model_arr = map_func.convert_to_array(self.model, self.model_iden,
                                                   self.map_var, self.map_tgt,
                                                   self.map_sep
                                                   )

    def _compute_pca(self) -> None:
        """full PCA analysis pipeiline"""

        # fit all pcas
        self.pca_objects = {
            "centered" : {
                "human": self.fit_pca(self.human_arr, self.n_comps, self.bs_seed, center=True),
                "model": self.fit_pca(self.model_arr, self.n_comps, self.bs_seed, center=True),
                "S_human": self.fit_pca(self.human_arr, self.n_comps, self.bs_seed, center=True, shuffle=True),
                "S_model": self.fit_pca(self.model_arr, self.n_comps, self.bs_seed, center=True, shuffle=True),
            },
            "uncentered": {
                "human": self.fit_pca(self.human_arr, self.n_comps, self.bs_seed),
                "model": self.fit_pca(self.model_arr, self.n_comps, self.bs_seed),
                "S_human": self.fit_pca(self.human_arr, self.n_comps, self.bs_seed, shuffle=True),
                "S_model": self.fit_pca(self.model_arr, self.n_comps, self.bs_seed, shuffle=True),
            }
        }

        # projection
        self.project_all()

    def _compute_mds(self) -> None:
        """
        full MDS analysis pipeiline
        -----------------------------------------------------------------------
        Return:
        mds_results: np.array (size: n_conds x n_mets x 2 x n_subjs)
            MDS results
        """

        # combine map_sep variables into a single matrix
        merge_arr = np.concatenate([self.human_arr, self.model_arr], axis=2)
        merge_arr = einops.rearrange(merge_arr, 'a b c d -> b c (a d)')

        # perform mds
        results = np.empty((merge_arr.shape[0], merge_arr.shape[1], 2))
        for i in range(merge_arr.shape[0]):
            mds = MDS(n_components=2, dissimilarity='precomputed', 
                      random_state=self.bs_seed
                      )
            corr_mat = map_func.compute_full_corr_matrix(
                merge_arr[i, :, :],
                merge_arr[i, :, :]
            )
            results[i,...] = mds.fit_transform(1 - corr_mat)

        # reshape results to split human and model
        human, model = np.split(results, 2, axis=1)
        results = np.stack((human, model), axis = 1)
        self.mds_results = results

    def plot_mds(self) -> None:
        """plot mds results"""

        n_metrics = self.mds_results.shape[0]

        plt.clf()
        fig, ax = plt.subplots(1, n_metrics, figsize=(n_metrics * 3, 3))
        colors = plt.cm.get_cmap('Dark2', 8)

        for i in range(n_metrics):
            ax[i].scatter(self.mds_results[i, 0, :, 0], self.mds_results[i, 0, :, 1], 
                          color=colors(0), label='Human', alpha=0.8, s=10
                          )
            ax[i].scatter(self.mds_results[i, 1, :, 0], self.mds_results[i, 1, :, 1],
                         color=colors(1), label='Model', alpha=0.8, s=10
                          )
            ax[i].set_title(f'{self.map_var[i]}', fontsize=14, fontweight='bold')
            ax[i].set_xlabel('MDS 1', fontsize=12)
            ax[i].set_ylabel('MDS 2', fontsize=12)
            ax[i].legend(fontsize=10)
    
        plt.suptitle('MDS results for each metric', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig_path = f'{self.graph_path}/DimsMDS.png'
        plt.savefig(fig_path, dpi=384)
        plt.close()
        print(fig_path)

    @staticmethod
    def _compute_var(data, result):
        return np.var(result, axis = 1) / np.sum(np.var(data, axis = 0))

    @staticmethod
    def _scale_data(data, scaler, cent, shuffle) -> np.array:
        # center and shuffle if needed
        if cent:
            data = map_func.center_to_zero(data)
        if shuffle:
            data = map_func.shuffle_image_order(data)
        return scaler.transform(data)

    def _plot_pca_common(self, center, pca_obj, plot_type='cumulative'):
        proj_arr = ['P_human', 'P_model', 'P_S_human', 'P_S_model']
        labels = ['Human', 'Model', 'Shuffled Human', 'Shuffled Model']
        proj_data_arr = [self.human_arr, self.model_arr, self.human_arr, self.model_arr]
        shuffle_array = [False, False, True, True]

        n_conds, n_metrics, _, _ = self.human_arr.shape
        plt.clf()
        fig, ax = plt.subplots(n_conds, n_metrics, figsize=(6, 4))
        colors = plt.cm.get_cmap('Dark2', 8)
        x_axis = np.arange(0, self.n_comps)

        # Ensure ax is always a 2D array
        if n_conds == 1:
            ax = np.array([ax])
        if n_metrics == 1:
            ax = np.array([ax]).T

        for i in range(n_conds):
            for j in range(n_metrics):
                for k, proj in enumerate(proj_arr):
                    # get the projection results
                    scaler_objs = self.pca_objects.item()[center][pca_obj]['scaler']
                    data = self._scale_data(proj_data_arr[k][i, j], scaler_objs[(i, j)],
                                            cent=(center == 'centered'),
                                            shuffle=shuffle_array[k]
                                            )
                    proj_results = self.pca_results.item()[center][pca_obj][proj]

                    # compute explained variance
                    explained_variance = self._compute_var(data, proj_results[i, j])

                    if plot_type == 'cumulative':
                        # plot cumulative explained variance
                        ax[i, j].plot(x_axis * 5,
                                      np.cumsum(explained_variance),
                                      color=colors(k), lw=2, label=labels[k]
                                      )
                        ax[i, j].set_ylim(0, 1)

                    else:
                        # plot explained variance as bars
                        buffer = [-1.5, -0.5, 0.5, 1.5]
                        ax[i, j].bar(x_axis * 5 + buffer[k], explained_variance,
                                     color=colors(k), alpha=0.5, label=labels[k],
                                     width=1
                                     )

                    ax[i, j].legend(fontsize=6)
                    ax[i, j].set_xlabel('PCA Components', fontsize=6, fontweight='bold')
                    ax[i, j].set_ylabel('Explained Variance', fontsize=6, fontweight='bold')
                    ax[i, j].set_xticks(x_axis * 5, x_axis + 1, fontsize=6)
                    ax[i, j].tick_params(axis='y', labelsize=8)
                    ax[i, j].set_title(f"Cond {i}, {self.map_var[j]}", fontsize=10, fontweight='bold')

        plt.suptitle(f'{pca_obj.capitalize()} PCA Projection ({center})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if plot_type == 'cumulative':
            fig_path = f'{self.graph_path}/DimsPCA_{center}_{pca_obj}_cumulative_plot.png'
        else:
            fig_path = f'{self.graph_path}/DimsPCA_{center}_{pca_obj}_explained_var_plot.png'
        plt.savefig(fig_path, dpi=384)
        plt.close()
        print(fig_path)

    def plot_pca_explained_var(self) -> None:
        """ plot pca results """
        self._convert_data_array()
        center_arr = ['centered', 'uncentered']
        type_arr = ['human', 'model']
        for i, center in enumerate(center_arr):
            for j, pca_obj in enumerate(type_arr):
                self._plot_pca_common(center, pca_obj, plot_type='cumulative')
                self._plot_pca_common(center, pca_obj, plot_type='non-cumulative')

    @staticmethod
    def fit_pca(arr, n_comps, seed, center=False, shuffle=False) -> dict:
        """Function to fit PCA and return scaler and pca objects"""

        # center and shuffle if needed
        if center:
            arr = map_func.center_to_zero(arr)
        if shuffle:
            arr = map_func.shuffle_image_order(arr)

        # initiate objects
        scaler_objs = {}
        pca_objs = {}
        scaled_arr = np.empty_like(arr)

        # fit pca for each separate map (axis 0) and metrics (axis 1)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                scaler = StandardScaler()
                pca = PCA(n_components=n_comps, random_state=seed)

                scaled_arr[i,j] = scaler.fit_transform(arr[i,j])
                pca.fit(scaled_arr[i,j])

                scaler_objs[(i,j)] = scaler
                pca_objs[(i,j)] = pca

        # return objects
        return {
            "scaler": scaler_objs,
            "pca": pca_objs,
        }

    def project_all(self) -> None:
        """
        Project data for all center variation, human, and model PCA
        -----------------------------------------------------------------------
        Return:
        pca_results: dict
            results of all projections
            1st [] : Center Variation (centered, uncentered)
            2nd [] : Source of PCA Components (human, model)
            3rd [] : Project to (P_human, P_model, P_S_human, P_S_model)
        """

        # initiate
        n_conds = self.human_arr.shape[0]
        n_mets = self.human_arr.shape[1]
        center_dict = ['centered', 'uncentered']
        type_dict = ['human', 'model']
        all_pcas = self.pca_objects
        self.pca_results = {}

        # loop to run all projection
        for c, c_dict in enumerate(center_dict):
            center = True if c_dict == 'centered' else False
            self.pca_results[c_dict] = {}
            for t, t_dict in enumerate(type_dict):
                scaler = all_pcas[c_dict][t_dict]['scaler']
                pca = all_pcas[c_dict][t_dict]['pca']

                self.pca_results[c_dict][t_dict] = {
                    'P_human': self._project(data = self.human_arr,
                                             pca = pca,
                                             scaler = scaler,
                                             center=center,
                                             ),
                    'P_model': self._project(data = self.model_arr,
                                             pca = pca,
                                             scaler = scaler,
                                             center=center,
                                             ),
                    'P_S_human': self._project(data = self.human_arr,
                                               pca = pca,
                                               scaler = scaler,
                                               center=center,
                                               shuffle=True
                                               ),
                    'P_S_model': self._project(data = self.model_arr,
                                               pca = pca,
                                               scaler = scaler,
                                               center=center,
                                               shuffle=True
                                               ),
                }

    @staticmethod
    def _project(data, pca, scaler, center=False, shuffle=False) -> np.array:
        """
        Function to project data onto PCA components
        -----------------------------------------------------------------------
        Return:
        results: np.array (size: n_conds x n_mets x n_comps x n_subjs)
            projection results

        """

        # center and shuffle if needed
        if center:
            data = map_func.center_to_zero(data)
        if shuffle:
            data = map_func.shuffle_image_order(data)

        # initate results
        n_conds = data.shape[0]
        n_mets = data.shape[1]
        n_comps = pca[(0,0)].components_.shape[0]
        n_subjs = data.shape[2]
        results = np.empty((n_conds, n_mets, n_comps, n_subjs))

        # project
        for i in range(n_conds):
            for j in range(n_mets):
                results[i,j] = pca[(i,j)].components_ @ scaler[(i,j)].transform(data[i,j,:,:]).T
        return results
