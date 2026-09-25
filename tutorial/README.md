# Tutorial: Using IndiMap for analyzing and mapping individual differences

This tutorial will guide you through the process of using the `IndiMap` library to analyze and map individual differences.

---

# Contents
- [Installation](#installation)
- [Quick Start](#quick-start-examplepy)
- [Data Requirements](#data-requirements)
- [API](#api)

---


# Installation
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
pip install git+https://github.com/herrickfung/indimap.git@v0.1.3
```

### 4. Run the example
```bash
python3 tutorial/example.py
```
Results are saved to `tutorial/indimap_results/` and plots to `tutorial/indimap_plots/`.

---


# Quick Start: `example.py`

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
        'subj_data': pd.read_csv('tutorial/data/human.csv'),
        'inst_data': pd.read_csv('tutorial/data/rtnet.csv'),
        'subj_column_name': 'subj',
        'inst_column_name': 'inst',
        'stim_column_name': 'stim',
        'resp_column_name': 'resp',
        'map_variables': ['acc', 'conf', 'rt'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'map_confusion': False,
        'map_category': False,
        'bootstrap_iterations': 10,
        'bootstrap_seed': 42,
        'nComp_PCA': 10,
        'output_path': 'tutorial/indimap_results',
        'graph_path': 'tutorial/indimap_plots'
    }
    ```
    **Details:**
    - `task_name`: Name of the task for you to identify the object (Default: `'Task'`).
    - `model_name`: Name of the model for you to identify the object (Default: `'Model'`).
    - `subj_data` (required): DataFrame containing human data, loaded from `tutorial/data/human.csv` in this example.
    - `inst_data` (required): DataFrame containing model data, loaded from `tutorial/data/rtnet.csv` in this example.
    - `subj_column_name`: Column name for subjects in the subject data (Default: `'subj'`).
    - `inst_column_name`: Column name for instances in the instance data (Default: `'inst'`).
    - `stim_column_name`: Column name for stimulus categories, used only when `map_category=True` or `map_confusion=True` (Default: `'stim'`).
    - `resp_column_name`: Column name for responses, used only when `map_confusion=True` (Default: `'resp'`).
    - `map_variables`: List of variables to map. Make sure that the variable name is consistent in both human and model dataset (e.g., accuracy, reaction time, confidence). The first variable is treated as accuracy (see [Data Requirements](#data-requirements)) (Default: `['acc', 'conf']`).
    - `map_together` (required): Column name to map together. Make sure that the variable name is consistent in both human and model dataset (e.g., image index, stimulus index).
    - `map_separate` (required): Column name to map separately. Make sure that the variable name is consistent in both human and model dataset. If you do not wish to map any variables separately, create a dummy column with a single value in all rows to ensure the code runs correctly (e.g., `'cond'`). The dummy column acts as a placeholder and does not affect the mapping process.
    - `map_confusion`: If `True`, additionally maps the confusion matrix (stimulus x response) as an extra metric named `confuse_mat` in CorrMap, RankMap, and TopMap. Requires `stim_column_name` and `resp_column_name` columns in both datasets. Cannot be combined with `map_category=True` (Default: `False`).
    - `map_category`: If `True`, additionally computes CorrMap and RankMap using splits of stimulus categories (from `stim_column_name`) instead of random splits of images. Retrieve these results with `split_by='cate'` (Default: `False`).
    - `bootstrap_iterations`: Number of bootstrap iterations (Default: `1000`).
    - `bootstrap_seed`: Seed for bootstrap sampling (Default: `42`).
    - `nComp_PCA`: Number of principal components for PCA (Default: `10`).
    - `output_path`: Path to save the results (Default: `'IndiMap_Result'`).
    - `graph_path`: Path to save the plots (Default: `'IndiMap_Plots'`).
    <br>

3. **Initialize the `IndiMap` object:**
    ```python
    mnist_rtnet = IndiMap(config)
    ```
    If you want to ensure the configuration is loaded correctly, you can check by:
    ```python
    print(mnist_rtnet)
    ```
    This also shows which analyses already have saved results in `output_path`.

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
    Results are saved to `output_path` (`CorrMap_results.npz`, `RankMap_results.npz`, `TopMap_results.npz`, `DimsMap_results.npz`, `PredMap_results.npz`). Set `load_exists=True` to load these saved results instead of recomputing.

    RankMap and TopMap reuse the correlation maps in `CorrMap_results.npz`, and compute CorrMap first if it does not exist.


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
    Plots are saved to `graph_path`. Plotting requires at least two `map_variables`; with a single variable, compute and retrieve results without plotting.

6. **Retrieve quantitative results:**
    Every getter returns a named tuple with a `dims` field describing each axis and a `mat` field holding the results:
    ```python
    results = mnist_rtnet.get_corr_results('subj', 'inst', target='subj', btw='split')
    print(results.dims)       # Dimensions: bootstrap iterations x metrics x subj
    print(results.mat.shape)  # (10, 3, 60)
    ```
    See [API](#api).

---


# Data Requirements
- Both datasets are in long format (e.g., one row per trial), with the same column names for `map_variables`, `map_together`, and `map_separate`. Rows are averaged within each individual, `map_separate`, and `map_together` value before mapping.
- Missing individual x item values are imputed with the mean of the other individuals.
- The first variable in `map_variables` is treated as accuracy. A warning is printed for any individual with average accuracy above 95% or below 5%. Human subjects above 99% or below 1% raise an error and must be removed; instances above 99% or below 1% are removed automatically from CorrMap.
- The number of subjects and instances can differ, except in PredMap, which requires at least as many instances as subjects. If there are more instances, only the first N instances (sorted by `inst_column_name`) are used as predictors, where N is the number of subjects.

---

# API

**Common parameters**
- `map_from`, `map_to`: The pair of response matrices to map. Available pairs are `('subj', 'inst')`, `('subj', 'subj')`, and `('inst', 'inst')`. For `('subj', 'subj')` and `('inst', 'inst')`, self-correlations are removed, so the target dimension has N - 1 entries.
- `btw`: `"split"` for consistency between two split-halves of images, or `"var"` for consistency between pairs of metrics. Metric pairs are ordered as all combinations of `map_variables` (e.g., `acc-conf`, `acc-rt`, `conf-rt`).
- `split_by`: `"rand"` for random splits of images (Default), or `"cate"` for splits of stimulus categories. `"cate"` requires `map_category=True` and is only available for `('subj', 'inst')` and `('subj', 'subj')`.

If a result is not computed or not available, `mat` is `None`.

**Methods**
- [compute_all()](#compute_allload_existsfalse): Compute all analyses.
- [plot_all()](#plot_all): Generate preliminary plots for quick and simple check.
- [get_corr_map()](#get_corr_mapmap_from-map_to-split_byrand): Retrieve the individual-level similarity matrix.
- [get_corr_results()](#get_corr_resultsmap_from-map_to-target-btw-split_byrand): Retrieve correlational mapping consistency results.
- [get_rank_results()](#get_rank_resultsmap_from-map_to-btw-split_byrand): Retrieve rank mapping consistency results.
- [get_top_map()](#get_top_mapmap_from-map_to): Retrieve the correlation map retaining only the best-mapped target.
- [get_top_ct()](#get_top_ctmap_from-map_to): Retrieve counts of how often each target is best-mapped.
- [get_top_corr()](#get_top_corrmap_from-map_to): Retrieve correlation values of the best-mapped target for each source.
- [get_top_results()](#get_top_resultsmap_from-map_to-btw-corr_on): Retrieve correlational mapping results considering only the best-mapped target.
- [get_top_expo()](#get_top_expomap_from-map_to): Retrieve exponential distribution fit results for best-mapped targets.
- [get_top_iden()](#get_top_idenmap_from-map_to-target): Retrieve identifiability results of the best-mapped target.
- [get_mds()](#get_mds): Retrieve Multi-Dimensional Scaling (MDS) results.
- [get_pca_results()](#get_pca_resultsproj_to): Retrieve split-half PCA results.
- [get_pred_results()](#get_pred_resultsby-using-within_metric): Retrieve prediction results based on specified parameters.

---


### `compute_all(load_exists=False)`
Compute all analyses.

**Parameters**  
- `load_exists` (`bool`, optional): If `True`, loads existing results from `output_path` instead of recomputing. Defaults to `False`.

**Notes**  
- To compute specific maps exclusively, replace `_all` with `_corr`, `_rank`, `_top`, `_dims`, or `_pred`.  
- This function may take a while depending on the dataset size.

---

### `plot_all()`
Generate preliminary plots for quick and simple check.

**Notes**  
- To plot specific maps exclusively, replace `_all` with `_corr`, `_rank`, `_top`, `_dims`, or `_pred`.
- Plots only show results from random splits of images (`split_by='rand'`).

---

### `get_corr_map(map_from, map_to, split_by='rand')`
Retrieve the individual-level similarity matrix for a specified pair of response matrices. Each entry is the correlation between two individuals' responses across images, averaged across `map_separate` conditions.

**Parameters**  
- `map_from` (`str`): Source response matrix, `"subj"` or `"inst"`.  
- `map_to` (`str`): Target response matrix, `"subj"` or `"inst"`.
- `split_by` (`str`, optional): `"rand"` or `"cate"`. Defaults to `"rand"`.

**Returns**  
- `CorrelationMapping` named tuple with fields:  
  - `dims` (`str`): `bootstrap iterations x split-half x metrics x map_from x map_to`. With `split_by='cate'`, the first dimension is category splits.  
  - `mat` (`np.ndarray` or `None`): Correlation matrix if available.

---

### `get_corr_results(map_from, map_to, target, btw, split_by='rand')`
Retrieve correlational mapping consistency results.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).  
- `target` (`str`): `"subj"`, `"inst"`, `"subj_gp"`, or `"inst_gp"`. `"subj"` refers to the rows (`map_from`) and `"inst"` refers to the columns (`map_to`) of the correlation map. `subj` and `inst` refer to consistency within the same target subject/instance. `subj_gp` and `inst_gp` refer to consistency across the target subject/instance and all other subjects/instances excluding the target subject/instance.
- `btw` (`str`): Between-group comparison, `"split"` or `"var"`.
- `split_by` (`str`, optional): `"rand"` or `"cate"`. Defaults to `"rand"`.

**Returns**  
- `CorrelationResults` named tuple with fields:  
  - `dims` (`str`): `bootstrap iterations x metrics x target` for `btw='split'`, or `bootstrap iterations x metric pairs x target` for `btw='var'`.  
  - `mat` (`np.ndarray` or `None`): Correlation results matrix.

---

### `get_rank_results(map_from, map_to, btw, split_by='rand')`
Retrieve rank mapping consistency results, measured as the sum of ranked correlation differences. Higher values indicate a more consistent ranking of targets.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).  
- `btw` (`str`): Between-group comparison, `"split"` or `"var"`.
- `split_by` (`str`, optional): `"rand"` or `"cate"`. Defaults to `"rand"`.

**Returns**  
- `RankResults` named tuple with fields:  
  - `dims` (`str`): `bootstrap iterations x metrics` for `btw='split'`, or `bootstrap iterations x metric pairs` for `btw='var'`.  
  - `mat` (`np.ndarray` or `None`): Rank results matrix.

---

### `get_top_map(map_from, map_to)`
Retrieve the correlation map retaining only the best-mapped target for each source. All other entries are `NaN`.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).

**Returns**  
- `TopMapping` named tuple with fields:  
  - `dims` (`str`): `bootstrap iterations x split-half x metrics x map_from x map_to`.  
  - `mat` (`np.ndarray` or `None`): Top correlation map.

---

### `get_top_ct(map_from, map_to)`
Retrieve the number of sources for which each target is the best-mapped target.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).

**Returns**  
- `TopCount` named tuple with fields:  
  - `dims` (`str`): `bootstrap iterations x split-half x metrics x map_to`.  
  - `mat` (`np.ndarray` or `None`): Count matrix.

---

### `get_top_corr(map_from, map_to)`
Retrieve correlation values of the best-mapped target for each source.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).

**Returns**  
- `TopCorr` named tuple with fields:  
  - `dims` (`str`): `bootstrap iterations x split-half x metrics x map_from`.  
  - `mat` (`np.ndarray` or `None`): Correlation values matrix.

---

### `get_top_results(map_from, map_to, btw, corr_on)`
Retrieve correlational mapping results considering only the best-mapped target.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).
- `btw` (`str`): Between-group comparison (`"split"` or `"var"`).  
- `corr_on` (`str`): Correlate on the counts from `get_top_ct()` (`"ct"`) or the correlation values from `get_top_corr()` (`"corr"`).

**Returns**  
- `TopResults` named tuple with fields:  
  - `dims` (`str`): `bootstrap iterations x metrics` for `btw='split'`, or `bootstrap iterations x metric pairs` for `btw='var'`.  
  - `mat` (`np.ndarray` or `None`): Top results matrix.

---

### `get_top_expo(map_from, map_to)`
Retrieve exponential distribution fit results for best-mapped targets. The sorted, normalized counts from `get_top_ct()` are fitted with `a * exp(b * x)`.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).

**Returns**  
- `TopExpo` named tuple with fields:  
  - `dims` (`str`): `(bootstrap iterations x split-half) x metrics x intercept/slope`.  
  - `mat` (`np.ndarray` or `None`): Intercept (`a`) and slope (`b`) of the exponential fit. `NaN` if the fit fails.

---

### `get_top_iden(map_from, map_to, target)`
Retrieve identifiability results of the best-mapped target. The best-mapped target of each source is identified in the first split-half of images, and its correlation is evaluated in the second split-half.

**Parameters**  
- `map_from` (`str`): Source response matrix (`"subj"` or `"inst"`).  
- `map_to` (`str`): Target response matrix (`"subj"` or `"inst"`).
- `target` (`str`): `"pair"` for the correlation with the best-mapped target, or `"gp"` for the mean correlation with all other targets.

**Returns**  
- `TopIden` named tuple with fields:  
  - `dims` (`str`): `bootstrap iterations x metrics x map_from`.  
  - `mat` (`np.ndarray` or `None`): Identifiability results matrix.

---

### `get_mds()`
Retrieve Multi-Dimensional Scaling (MDS) results. Subjects and instances are embedded together in two dimensions, using 1 - correlation across all images and conditions as the distance. MDS results may differ slightly across scikit-learn versions.

**Returns**  
- `MDS_Results` named tuple with fields:  
  - `dims` (`str`): `metrics x human/model x N x 2 MDS dimensions`, where N is the larger of the number of subjects and instances.  
  - `mat` (`np.ndarray` or `None`): MDS results matrix. If the number of subjects and instances differ, the smaller group is padded with `NaN`.

---

### `get_pca_results(proj_to)`
Retrieve split-half PCA results. PCA is fitted on a random half of the subjects, then the data specified by `proj_to` is projected onto the PCA components.

**Parameters**  
- `proj_to` (`str`): Data projected onto the PCA components:
  - `"same_human"`: The half of subjects used for fitting.
  - `"diff_human"`: The held-out half of subjects.
  - `"scrm_human"`: The half of subjects used for fitting, with image order scrambled.
  - `"model"`: A random half of the instances.

**Returns**  
- `PCA_Results` named tuple with fields:  
  - `dims` (`str`): `conditions x metrics x PCA components` for `"same_human"` and `"diff_human"`, or `bootstrap iterations x conditions x metrics x PCA components` for `"scrm_human"` and `"model"`.  
  - `mat` (`np.ndarray` or `None`): Explained variance of each PCA component.

---

### `get_pred_results(by, using, within_metric)`
Retrieve prediction results based on specified parameters. Each human subject's responses to a held-out half of images are predicted from other individuals, and prediction accuracy is the correlation between predicted and actual responses.

**Parameters**  
- `by` (`str`): Predict from other subjects (`"subj"`) or from instances (`"inst"`).  
- `using` (`str`): Prediction method:
  - `"rand"`: A random individual.
  - `"avg"`: Average of all individuals.
  - `"corr"`: Average of all individuals, weighted by their CorrMap similarity to the predicted subject.
- `within_metric` (`bool`): `True` for within-metric predictions (e.g., `acc` predicts `acc`), `False` for across-metric predictions (e.g., `acc` predicts `rt`).

**Returns**  
- `PredResults` named tuple with fields:  
  - `dims` (`str`): `bootstrap iterations x metrics x subj` for within-metric predictions, or `bootstrap iterations x metric pairs x subj` for across-metric predictions.  
  - `metric` (`list`): Metric ordering. For across-metric predictions, each pair is `(predictor metric, predicted metric)`.  
  - `mat` (`np.ndarray` or `None`): Prediction results.
