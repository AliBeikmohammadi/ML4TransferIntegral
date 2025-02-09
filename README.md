# ML4TransferIntegral

## Overview
This repository contains the code accompanying the paper:

**"From Solvent Baths to Charge Paths: Deciphering Conductivity in PEDOT:TOS guided by Machine Learning"**

Authors: Najmeh Zahabi, Ioannis Petsagkourakis, Juan Felipe Franco Gonzalez, Nicolas Rolland, Ali Beikmohammadi, Xianjie Liu, Mats Fahlman, Eleni Pavlopoulou, and Igor Zozoulenko.

## Description
This project employs machine learning techniques to analyze and predict the conductivity behavior of PEDOT:TOS, a conductive polymer. The code is structured to:
- Process experimental and simulated datasets.
- Apply machine learning models to uncover correlations between structural properties and electrical conductivity.
- Generate insights into charge transport mechanisms.

## Repository Structure
```
|-- ML-GitHub.ipynb        # Jupyter Notebook with the main analysis pipeline
|-- README.md              # This file
```

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/AliBeikmohammadi/ML4TransferIntegral.git
   cd your-repo
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

## Usage
Open the Jupyter Notebook and run the analysis:
```bash
jupyter notebook ML-GitHub.ipynb
```
Follow the steps in the notebook to preprocess data, train models, and interpret results.

## Dependencies
Ensure you have the required Python packages installed. The main dependencies include:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

All required dependencies are listed in `ML-GitHub.ipynb`.

## Citation
If you use this code in your research, please cite our paper:
```
@article{your_paper_reference,
  author = {Najmeh Zahabi and others},
  title = {From Solvent Baths to Charge Paths: Deciphering Conductivity in PEDOT:TOS guided by Machine Learning},
  journal = {},
  year = {2025}
}
```

## License
This project is released under the MIT License. See `LICENSE` for details.

## Contact
For any questions or issues, please contact the authors or open an issue in this repository.

