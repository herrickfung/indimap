# Tutorial: Using IndiMap for analyzing and mapping individual differences

This tutorial will guide you through the process of using the `IndiMap` library to analyzing and mapping individual differences.

---

## Contents
- [Installation](#installation)
- [Quick Start](#quick-start-examplepy)
- [API](#api)

---


## Installation
To set up this tutorial locally, follow these steps:

### 1. Clone the repository:
Clone the repository to your local machine and navigate into the project folder:
```bash
git clone https://github.com/herrickfung/indimap.git
cd indimap
```

### 2. Set up a Python environment (optional but recommended)
```bash
python3 -m venv ./venv/
source ./venv/bin/activate
```

### 3. Install IndiMap
```bash
pip install .
# or install from GitHub
pip install git+https://github.com/herrickfung/indimap.git@v0.1.1
```

### 4. Run the example
```bash
python3 tutorial/example.py
```

---


## Quick Start: `example.py`

1. **Import the necessary libraries:**
    ```python
    from indimap import IndiMap
    import pandas as pd
    ```

2. **Prepare the configuration dictionary:**
    ```python
    config = {
        'task_name': 'MNIST',
        'model_name': 'RTNet',
        'subj_data': pd.read_csv('data/human.csv'),
        'inst_data': pd.read_csv('data/rtnet.csv'),
        'subj_column_name': 'subj',
        'inst_column_name': 'inst',
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'bootstrap_iterations': 10,
        'bootstrap_seed': 42,
        'nComp_PCA': 10,
        'output_path': 'indimap_results',
        'graph_path': 'indimap_plots'
    }
    ```
    **Details:**
    - `task_name`: Name of the task for you to identify the object (Default: 'Task').
    - `model_name`: Name of the model for you to identify the object (Default: 'Model').
    - `subj_data`: DataFrame containing human data, loaded from 'data/human.csv' in this example.
    - `inst_data`: DataFrame containing model data, loaded from 'data/rtnet.csv' in this example.
    - `subj_column_name`: Column name for subjects in the subject data (Default: 'subj').
    - `inst_column_name`: Column name for instances in the instance data (Default: 'inst').
    - `map_variables`: List of variables to map. Make sure that the variable name is consistent in both human and model dataset (e.g., accuracy, reaction time, confidence) (Default: ['acc', 'conf']).
    - `map_together`: Column name to map together. Make sure that the variable name is consistent in both human and model dataset (e.g., image index, stimulus index).
    - `map_separate`: Column name to map separately. Make sure that the variable name is consistent in both human and model dataset. If you do not wish the map any variables separately, create a dummy column with a single value in all rows to ensure the code runs correctly (e.g., 'cond'). The dummy column acts asa a placeholder and does not affect the mapping process.
    - `bootstrap_iterations`: Number of bootstrap iterations (Default: 1000). 
    - `bootstrap_seed`: Seed for bootstrap sampling (Default: 42).
    - `nComp_PCA`: Number of principal components for PCA (Default: 10).
    - `output_path`: Path to save the results.
    - `graph_path`: Path to save the plots.
    <br>

3. **Initialize the `IndiMap` object:**
    ```python
    mnist_rtnet = IndiMap(config)
    ```
    If you want to ensure the configuration is loaded correctly, you can check by:
    ```python
    print(mnist_rtnet)
    ```

4. **Compute all analyses:**
    ```python
    mnist_rtnet.compute_all(load_exists=False)
    ```
    If you wish to compute specific analysis, replace _all with _corr, _rank, _top, _dims, _pred depending on your needs:
    ```python
    mnist_rtnet.compute_corr(load_exists=False)
    mnist_rtnet.compute_rank(load_exists=False)
    mnist_rtnet.compute_top(load_exists=False)
    mnist_rtnet.compute_dims(load_exists=False)
    mnist_rtnet.compute_pred(load_exists=False)
    ```


5. **Generate preliminary plots:**
    ```python
    mnist_rtnet.plot_all()
    ```
    If you wish to plot specific analysis or if you did not compute all analysis, replace _all with _corr, _rank, _top, _dims, _pred depending on your needs:
    ```python
    mnist_rtnet.plot_corr()
    mnist_rtnet.plot_rank()
    mnist_rtnet.plot_top()
    mnist_rtnet.plot_dims()
    mnist_rtnet.plot_pred()
    ```

6. **Retrieve quantitative results:**
    See [API](#api).

---

## API
- [compute_all()](#compute_allload_existsfalse): Compute all analyses.
- [plot_all()](#plot_all): Generate preliminary plots for quick and simple check.
- [get_corr_map()](#get_corr_mapmap_from-map_to): Retrieve the individual-level similarity matrix.
- [get_corr_results()](#get_corr_resultsmap_from-map_to-target-btw): Retrieve correlational mapping consistency results.
- [get_rank_results()](#get_rank_resultsmap_from-map_to-btw): Retrieve rank mapping consistency results.
- [get_top_map()](#get_top_mapmap_from-map_to): Retrieve the correlation map retaining only the best-mapped target.
- [get_top_ct()](#get_top_ctmap_from-map_to): Retrieve counts of the best-mapped target for each source.
- [get_top_corr()](#get_top_corrmap_from-map_to): Retrieve correlation values of the best-mapped target for each source.
- [get_top_results()](#get_top_resultsmap_from-map_to-btw-corr_on): Retrieve correlational mapping results considering only the best-mapped target.
- [get_top_expo()](#get_top_expomap_from-map_to): Retrieve exponential distribution fit results for best-mapped targets.
- [get_mds()](#get_mds): Retrieve Multi-Dimensional Scaling (MDS) results.
- [get_pca_results()](#get_pca_resultsfit_on-proj_to-center-scramble): Retrieve PCA results based on specified parameters.
- [get_pred_results()](#get_pred_resultsby-using-within_metric): Retrieve prediction results based on specified parameters.

---


### `compute_all(load_exists=False)`
Compute all analyses.

**Parameters**  
- `load_exists` (`bool`, optional): If `True`, loads existing results instead of recomputing. Defaults to `False`.

**Notes**  
- To compute specific maps exclusively, replace `_all` with `_corr`, `_rank`, `_top`, `_dims`, or `_pred`.  
- This function may take a while depending on the dataset size.

---

### `plot_all()`
Gernerate preliminary plots for quick and simple check.

**Notes**  
- To plot specific maps exclusively, replace `_all` with `_corr`, `_rank`, `_top`, `_dims`, or `_pred`.

---

### `get_corr_map(map_from, map_to)`
Retrieve the individual-level similarity matrix for a specified pair of response matrices.

**Parameters**  
- `map_from` (`str`): Source response matrix, `"subj"` or `"inst"`.  
- `map_to` (`str`): Target response matrix, `"subj"` or `"inst"`.

**Returns**  
- `CorrelationMapping` named tuple with fields:  
  - `dims` (`str`): Description of the correlation map dimensions.  
  - `mat` (`np.ndarray` or `None`): Correlation matrix if available.

---

### `get_corr_results(map_from, map_to, target, btw)`
Retrieve correlational mapping consistency results.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).  
- `target` (`str`): Target for between-group comparison (`"subj"`, `"inst"`, `subj_gp`, `inst_gp`), `subj` and `inst` refers to consistency within the same target subject/instance. `subj_gp` and `inst_gp` refers to consistency across the target subject/instance and all other subjects/instances excluding the target subject/instance.
- `btw` (`str`): Between-group comparison, `"split"` or `"var"`.

**Returns**  
- `CorrelationResults` named tuple with fields:  
  - `dims` (`str`): Description of dimensions.  
  - `mat` (`np.ndarray` or `None`): Correlation results matrix.

---

### `get_rank_results(map_from, map_to, btw)`
Retrieve rank mapping consistency results.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).  
- `btw` (`str`): Between-group comparison, `"split"` or `"var"`.

**Returns**  
- `RankResults` named tuple with fields:  
  - `dims` (`str`): Description of dimensions.  
  - `mat` (`np.ndarray` or `None`): Rank results matrix.

---

### `get_top_map(map_from, map_to)`
Retrieve the correlation map retaining only the best-mapped target.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).

**Returns**  
- `TopMapping` named tuple with fields:  
  - `dims` (`str`): Description of dimensions.  
  - `mat` (`np.ndarray` or `None`): Top correlation map.

---

### `get_top_ct(map_from, map_to)`
Retrieve counts of the best-mapped target for each source.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).

**Returns**  
- `TopCount` named tuple with fields:  
  - `dims` (`str`): Description of dimensions.  
  - `mat` (`np.ndarray` or `None`): Count matrix.

---

### `get_top_corr(map_from, map_to)`
Retrieve correlation values of the best-mapped target for each source.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).

**Returns**  
- `TopCorr` named tuple with fields:  
  - `dims` (`str`): Description of dimensions.  
  - `mat` (`np.ndarray` or `None`): Correlation values matrix.

---

### `get_top_results(map_from, map_to, btw, corr_on)`
Retrieve correlational mapping results considering only the best-mapped target.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).
- `btw` (`str`): Between-group comparison (`"split"` or `"var"`).  
- `corr_on` (`str`): Metric to correlate on (`"ct"` or `"corr"`).

**Returns**  
- `TopResults` named tuple with fields:  
  - `dims` (`str`): Description of dimensions.  
  - `mat` (`np.ndarray` or `None`): Top results matrix.

---

### `get_top_expo(map_from, map_to)`
Retrieve exponential distribution fit results for best-mapped targets.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).

**Returns**  
- `TopExpo` named tuple with fields:  
  - `dims` (`str`): Description of dimensions.  
  - `mat` (`np.ndarray` or `None`): Exponential slope/intercept results.

---

### `get_mds()`
Retrieve Multi-Dimensional Scaling (MDS) results.

**Returns**  
- `MDS_Results` named tuple with fields:  
  - `dims` (`str`): Description of dimensions.  
  - `mat` (`np.ndarray`): MDS results matrix.

---

### `get_pca_results(fit_on, proj_to, center, scramble)`
Retrieve PCA results based on specified parameters.

**Parameters**  
- `fit_on` (`str`): Dataset used to fit PCA (`"human"` or `"model"`).  
- `proj_to` (`str`): Projection target (`"human"` or `"model"`).  
- `center` (`bool`): Use centered PCA if `True`.  
- `scramble` (`bool`): Use scrambled projection if `True`.

**Returns**  
- `PCA_Results` named tuple with fields:  
  - `dims` (`str`): Description of dimensions.  
  - `mat` (`np.ndarray`): PCA results matrix.

---

### `get_pred_results(by, using, within_metric)`
Retrieve prediction results based on specified parameters.

**Parameters**  
- `by` (`str`): Perform prediction by this variable (`"subj"` or `"inst"`).  
- `using` (`str`): Prediction method (`"rand"`, `"avg"`, `"corr"`). 
- `within_metric` (`bool`): True for within-metric predictions, False for across-metric predictions.

**Returns**  
- `PredResults` named tuple with fields:  
  - `dims` (`str`): Description of prediction dimensions.  
  - `metric` (`list` or `str`): Metric ordering.  
  - `mat` (`np.ndarray` or other): Prediction results.
