# Meta-Learning for XGBoost
Final project for the course _Automated Machine Learning_ at LMU WS22/23.

Author: Dusan Urosevic, student MSc Data Science

### Setup environment

If using Python and _anaconda/miniconda_ already installed, the following command should setup the required environment:
```commandline
conda create --name <env> --file requirements_conda.txt
```

For a more general installation of dependencies:
```commandline
pip install -r requirements.txt
```

For the notebooks you need a Python 3 based kernel. If for some reason neither works, you need to install Python 3.8+ and the listed packages. 

### Packages used
* openml, pandas, xgboost - from original code skeleton 
* [scikit-learn](https://scikit-learn.org/stable/install.html) - for regression models used in the project
* [scikit-optimize](https://scikit-optimize.github.io/stable/install.html) - for the BayesSearchCV function that is used to perform Bayesian Optimisation
* [numpy](https://numpy.org/install/) - for working with matrices and sampling configurations
* [scipy](https://scipy.org/install/) - for function minimisation methods
* [matplotlib](https://matplotlib.org/stable/users/installing/index.html) - for visualisations of the results
* [pytorch](https://pytorch.org/get-started/locally/) - for defining a neural network architecture and working with tensors


### Contents
The project results and visualisations are given in the notebook [RESULTS.ipynb](https://github.com/dusan0098/AutoML-XGBoost/blob/main/python/RESULTS.ipynb). It contains examples of all methods 
presented in the project report and instructions on how to run them if you wish to retrain the models from scratch.

**IMPORTANT**: In order to run the code you need to **extract the entire contents** of the following [zip file](https://drive.google.com/drive/folders/1bdzU29C1DhGkdfWBCOirKvQarbvjUvJu?usp=sharing) to the **python** directory
of the project.

For more details on the implementations of individual methods:
* [Direct regression](https://github.com/dusan0098/AutoML-XGBoost/blob/main/python/compare_regressors.py)
* [Taskwise regression](https://github.com/dusan0098/AutoML-XGBoost/blob/main/python/taskRegression.py)
* [Joint regression](https://github.com/dusan0098/AutoML-XGBoost/blob/main/python/fullDataRegression.py)
* [Implementation of Fully connected network](https://github.com/dusan0098/AutoML-XGBoost/blob/main/python/FFN_regression.py)
* [Utility functions - for preparing data used by the regression models](https://github.com/dusan0098/AutoML-XGBoost/blob/main/python/project_utils.py)
* [Evaluating baseline performance](https://github.com/dusan0098/AutoML-XGBoost/blob/main/python/baseline.py)

**NOTE:** All other notebooks in this project were used to develop the methods in the listed .py files. There is no need to run them and doing so might overwrite some of the 
saved models/results.




