# Toy Flow Matching

A minimal implementation of the flow matching and rectified flow algorithms for synthetic data generation.

## Repo structure

```text
.gitignore      -> configure files to be ignore in git commits
LICENSE.md      -> license file 
data.py         -> functions to generate or load sample datasets
datasets/       -> bundled datasets
└── banana.csv  -> classic banana dataset
distances.py    -> Wasserstein distance computation
embedding.py    -> methods to embed high-dimensional data into 2D, for visualizations
generators.py   -> PyTorch datasets for online generation of data or synthetic data
images.ipynb    -> Notebook to test image generation flows
models.py       -> Neural network models, training algorithms and samples generations
plotting.py     -> Functions to generate visualizations, used in the notebooks
requirements.txt    -> packages required to run this repo
toy_data_supervised.ipynb       -> Notebook to test supervised generation flows over toy 2D data
toy_data_unsupervised.ipynb     -> Notebook to test unsupervised generation flows over toy 2D data
```

## How to get started with this repo

### 1. Clone the repository

```bash
git clone https://github.com/albarji/toy-flow-matching.git
cd toy-flow-matching
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The notebooks in the repository were saved with a Python kernel named `toy-flow-matching` and Python `3.14.3`. If you use a different supported Python version, make sure the pinned packages in `requirements.txt` install successfully for that interpreter.

### 4. Start with a notebook

For the shortest conceptual path, use this order:

1. **`toy_data_unsupervised.ipynb`** — learn the basic flow-matching pipeline on a 2D distribution, then explore rectification, reflow, distillation, and reverse trajectories.
2. **`toy_data_supervised.ipynb`** — extend the same ideas to label-conditioned flows.
3. **`images.ipynb`** — move to image-shaped data and the U-Net model.

## How to cite this repo

> Barbero Jiménez, Álvaro. *Toy Flow Matching*. GitHub repository, 2026. https://github.com/albarji/toy-flow-matching.

BibTeX entry:

```bibtex
@software{barbero_jimenez_toy_flow_matching,
  author  = {Barbero Jiménez, Álvaro},
  title   = {Toy Flow Matching},
  year    = {2026},
  url     = {https://github.com/albarji/toy-flow-matching}
}
```