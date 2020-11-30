Setup
=====

1. Install `conda` (e.g., `miniconda`, see [instructions here](https://docs.conda.io/en/latest/miniconda.html))
2. Create a new `conda` environment, e.g., `i2k_gunpowder` using the provided `conda` YAML file:
```
conda env create -n i2k_gunpowder --file conda.yaml
conda activate i2k_gunpowder
```
3. Install the `funlib` helper packages:
```
pip install -r requirements.txt
```
4. Run Jupyter
```
jupyter notebook
```
