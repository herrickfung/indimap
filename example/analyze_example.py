from indimap import IndiMap
import pandas as pd

human_data = pd.read_csv('data/human.csv')
rtnet_data = pd.read_csv('data/rtnet.csv')

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
}

mnist_rtnet = IndiMap(config)
mnist_rtnet.compute_all(load_exists = False)

