# ML4TransferIntegral

## Overview
This repository accompanies the paper:

**"From Solvent Baths to Charge Paths: Deciphering Conductivity in PEDOT:TOS guided by Machine Learning"**

Authors: Najmeh Zahabi, Ioannis Petsagkourakis, Nicolas Rolland, Ali Beikmohammadi, Xianjie Liu, Mats Fahlman, Eleni Pavlopoulou, and Igor Zozoulenko.

---

## Description

This project integrates **physics-based simulation** with **deep learning models** to understand and predict electronic properties of PEDOT:TOS, a conductive polymer.

It includes:
- Geometry-based preprocessing and input file generation.
- Post-processing of transfer integrals to compute hopping rates and mobility.
- A deep learning framework (CNN/GNN) for predicting HOMO energies from matrix-encoded descriptors.

---

## Repository Structure

```
|-- ML-GitHub.ipynb        # Deep learning pipeline using CNN/GNN for HOMO prediction
|-- Preprocessing.py       # Delaunay triangulation, dimers generation, sbatch scripts
|-- Mobility.py            # Physical model to compute hopping rates and mobility
|-- README.md              # Project overview and usage guide
|-- data/                  # Contains centers.dat and atomic coordinate files
|-- dimmers/               # Generated Gaussian input files for dimers (Auto-generated)
|-- sbatch/                # SLURM job scripts and launcher (Auto-generated)
|-- output/                # Transfer integral results and logs (Auto-generated)
|-- logs/                  # TensorBoard logs and model checkpoints (Auto-generated)
```

---

## Installation

```bash
git clone https://github.com/AliBeikmohammadi/ML4TransferIntegral.git
cd ML4TransferIntegral
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Workflow Overview

### 1. Preprocessing Step: Dimers and Job Preparation
```bash
python Preprocessing.py
```
- Uses Delaunay triangulation to find neighbors from `centers.dat`.
- Generates `.gjf` files for dimers.
- Creates `sbatch` scripts for running quantum chemical calculations.

### 2. Transfer Integral & Mobility Calculation
After running Gaussian and extracting HOMO/transfer integrals:
```bash
python Mobility.py
```
- Cleans transfer_integrals.csv.
- Adds inter-site distances (Rij) and hopping rates (Wij).
- Builds 3D hopping matrices.
- Iteratively solves for occupation probabilities and computes mobility in X, Y, Z.

---

## 3. Deep Learning Pipeline (Jupyter Notebook)

Launch:
```bash
jupyter notebook ML-GitHub.ipynb
```

This notebook allows you to:
- **Load CIP/CM `.npy` descriptors** and associated `transfer_integrals.csv` targets.
- **Filter and merge** data by condition and material.
- **Train GNN or CNN models** on single or multi-input descriptors (`CIP`, `CM`, or both).
- **Log and visualize results** using TensorBoard and matplotlib.
- **Evaluate performance** using RMSE, MAE, and R².
- **Predict and visualize individual sample performance.**

### Supported Features
- Automatic model architecture definition (configurable layer depths).
- Reproducible training with `SEED`.
- Model checkpointing and export (`model.pt`, `best_model.pt`).
- Hyperparameter-based logging structure.

---

## Example Configuration (Notebook)
```python
base_folder = './Data1/'
target = 'target_HOMO'  # or 'target_log_abs_HOMO'
input_type = 'Multi'    # Options: 'CIP', 'CM', 'Multi'
model_type = 'GNN'      # or 'CNN'
epochs = 1000
batch_size = 64
lr = 0.001
```

Then:
```python
train_model(train_loader, test_loader, model, optimizer, criterion, ...)
```

---

## Dependencies

Install via `pip install -r requirements.txt` or manually:
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `scipy`, `torch`, `torch-geometric`, `tensorboard`
- `jupyter`, `csv`, `os`, `json`

---

## Citation
If you use this code in your research, please cite our paper:
```
@article{Zahabi2025,
  author = {Najmeh Zahabi and others},
  title = {From Solvent Baths to Charge Paths: Deciphering Conductivity in PEDOT:TOS guided by Machine Learning},
  journal = {},
  year = {2025}
}
```

---

## License
Released under the **MIT License**.

---

## Contact

For questions, open an issue or contact the authors directly.

## Contact
For any questions or issues, please contact the authors or open an issue in this repository.

