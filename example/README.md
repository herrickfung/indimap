# Tutorial: Using IndiMap for analyzing and mapping individual differences

This tutorial will guide you through the process of using the `IndiMap` library to analyzing and mapping individual differences in human behavior to convolutional neural networks in perceptual decision-making tasks.

## Step-by-Step Guide

1. **Import the necessary libraries:**
    ```python
    from indimap import IndiMap
    import pandas as pd
    ```

2. **Prepare the configuration dictionary:**
    Please read the explanation below for details of the configuration dictionary.
    ```python
    config = {
        'task_name': 'MNIST',
        'model_name': 'RTNet',
        'subj_data': pd.read_csv('data/human.csv'),
        'inst_data': pd.read_csv('data/rtnet.csv'),
        'subj_column_name': 'subj',
        'inst_column_name': 'inst',
        'map_variables': ['acc', 'rt', 'conf'],
        'map_together': 'mnist_index',
        'map_separate': 'cond',
        'bootstrap_iterations': 10,
        'bootstrap_seed': 42,
        'nComp_PCA': 10,
        'output_path': 'indimap_results',
        'graph_path': 'indimap_plots'
    }
    ```

3. **Initialize the `IndiMap` object:**
    ```python
    mnist_rtnet = IndiMap(config)
    ```
    If you want to ensure the configuration is loaded correctly, you can check by:
    ```python
    print(mnist_rtnet)
    ```

4. **Compute all necessary mappings:**
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


5. **Generate and save the plots:**
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

## Explanation

- **Configuration Dictionary (`config`):**
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

- **Methods:**
    - `compute_all(load_exists=False)`: Computes all necessary mappings. Running this function may take a while (in general less than 15 minutes). If you wish to compute specific maps exclusively, replace _all with _corr, _rank, _top, _dims, _pred depending on your needs. Set `load_exists` to `True` if you want to load existing results.
    - `plot_all()`: Generates and saves all plots. If you wish to plot specific map exclusively, replace _all with _corr, _rank, _top, _dims, _pred depending on your needs.
