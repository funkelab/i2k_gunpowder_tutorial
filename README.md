Setup
=====

1. Install `conda` (e.g., `miniconda`, see [instructions here](https://docs.conda.io/en/latest/miniconda.html))
2. Create a new `conda` environment, e.g., `i2k_gunpowder` using the provided `conda` YAML file:
```
conda env create -n i2k_gunpowder --file environment.yaml
conda activate i2k_gunpowder
```
3. Install the `neuroglancer` helper script
```
pip install -r requirements.txt
```
