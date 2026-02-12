# IndiMap Library
This repository provides the IndiMap library, a Python library designed for analyzing and mapping individual differences in human and neural networks.

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
To install IndiMap,
```bash
pip install git+https://github.com/herrickfung/indimap.git@v0.1.1
```

If you're running on virtual environment, 
```bash
python3 -m venv ./venv/
source ./venv/bin/activate
pip install git+https://github.com/herrickfung/indimap.git@v0.1.1
```

## Usage
See [tutorial](/tutorial).

## Data
Human and neural network behavioral data used in the examples are retrieved from:

F. Rafiei, M. Shekhar, & D. Rahnev (2024). The neural network RTNet exhibits the signatures of human perceptual decision-making. *Nature Human behavior*, _8_(1), 1752-1770. [https://doi.org/10.1038/s41562-024-01914-8](https://doi.org/10.1038/s41562-024-01914-8)

## Citation
If you use this library in academic work, please cite:

Fung, H., Murty, N. A. R., & Rahnev, D. (2026). Individual differences in artificial neural networks capture individual differences in human behavior (p. 2026.02.10.705061). bioRxiv. [https://doi.org/10.64898/2026.02.10.705061](https://www.biorxiv.org/content/10.64898/2026.02.10.705061)

Fung, H., Murty, N. A. R., & Rahnev, D. (2025). Human-like individual differences emerge from random weight initializations in neural networks (p. 2025.10.25.684448). bioRxiv. [https://doi.org/10.1101/2025.10.25.684448](https://doi.org/10.1101/2025.10.25.684448)

```bibtex
@article{Fung2026IndividualDifferencesANN,
  title = {Individual differences in artificial neural networks capture individual differences in human behavior},
  author  = {Fung, Herrick and Murty, N. A. R. and Rahnev, Dobromir},
  journal = {bioRxiv},
  year    = {2026},
  pages   = {2026.02.10.705061},
  doi     = {10.64898/2026.02.10.705061},
  url     = {https://doi.org/10.64898/2026.02.10.705061}
}

@article{Fung2025HumanLikeID,
  title   = {Human-like individual differences emerge from random weight initializations in neural networks},
  author  = {Fung, Herrick and Murty, N. A. R. and Rahnev, Dobromir},
  journal = {bioRxiv},
  year    = {2025},
  pages   = {2025.10.25.684448},
  doi     = {10.1101/2025.10.25.684448},
  url     = {https://doi.org/10.1101/2025.10.25.684448}
}
```

## Enquiries
[Herrick Fung](mailto:herrickfung@gmail.com)
