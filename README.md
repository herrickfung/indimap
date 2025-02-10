# IndiMap Library
This repository provides the IndiMap library, a Python library designed for analyzing and mapping individual differences in human behavior onto convolutional neural networks in perceptual decision-making tasks.

## Python Dependencies
To use the IndiMap library, ensure the following Python dependencies are installed. The library is tested exclusively with these versions:

1. einops==0.8.0
2. numpy==1.23.5
3. pandas==1.5.3
4. scikit-learn==1.4.0
5. scipy==1.11.4
6. tqdm==4.66.1

## Installation
Clone the repository and run the following commands in the root directory:
```bash
git clone https://github.com/herrickfung/IndiMap.git
pip install -r requirements.txt
pip install .
```

If you're running on virtual enviornment, 
```bash
python3 -m venv ./venv/
source ./venv/bin/activate
git clone https://github.com/herrickfung/IndiMap.git
pip install -r requirements.txt
pip install .
```

## Usage
Please refer to the ```example``` directory for usage and details. To run: 

```bash
python3 analyze_example.py
python3 plot_exanmple.py
```

## Data
Human and neural network behavioral data used in the examples are retrieved from:

F. Rafiei, M. Shekhar, & D. Rahnev (2024). The neural network RTNet exhibits the signatures of human perceptual decision-making. *Nature Human behavior*, _8_(1), 1752-1770. [https://doi.org/10.1038/s41562-024-01914-8](https://doi.org/10.1038/s41562-024-01914-8)

## Citation
If you use this library in academic work, please cite:

XXX
