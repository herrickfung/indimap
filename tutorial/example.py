from indimap import IndiMap
from pathlib import Path
import pandas as pd

tutorial_dir = Path(__file__).parent

config = {
    'task_name': 'MNIST',
    'model_name': 'RTNet',
    'subj_data': pd.read_csv(tutorial_dir / 'data/human.csv'),
    'inst_data': pd.read_csv(tutorial_dir / 'data/rtnet.csv'),
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
    'output_path': tutorial_dir / 'indimap_results',
    'graph_path': tutorial_dir / 'indimap_plots'
}

mnist_rtnet = IndiMap(config)
print(mnist_rtnet)
mnist_rtnet.compute_all(load_exists = False)
mnist_rtnet.plot_all()
