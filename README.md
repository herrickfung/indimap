# IndiMap Library
This repository provides the IndiMap library, a Python library designed for analyzing and mapping individual differences in human behavior onto convolutional neural networks in perceptual decision-making tasks.

## Python Dependencies
The IndiMap library requires the following dependencies:

1. einops==0.8.0
1. matplotlib==3.6.3
1. numpy==1.23.5
1. pandas==1.5.3
1. scikit-learn==1.4.0
1. scipy==1.11.4
1. tqdm==4.66.1

## Installation
Clone the repository and run the following commands in the root directory:
```bash
git clone https://github.com/herrickfung/IndiMap.git
cd IndiMap
pip install .
```

If you're running on virtual environment, 
```bash
python3 -m venv ./venv/
source ./venv/bin/activate
git clone https://github.com/herrickfung/IndiMap.git
cd IndiMap
pip install .
```

## Usage
Please refer to the ```example``` directory for usage and details. To run: 

```bash
python3 analyze_example.py
```

## Data
Human and neural network behavioral data used in the examples are retrieved from:

F. Rafiei, M. Shekhar, & D. Rahnev (2024). The neural network RTNet exhibits the signatures of human perceptual decision-making. *Nature Human behavior*, _8_(1), 1752-1770. [https://doi.org/10.1038/s41562-024-01914-8](https://doi.org/10.1038/s41562-024-01914-8)

## Citation
If you use this library in academic work, please cite:

XXX
