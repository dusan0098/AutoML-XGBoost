#!/bin/bash

# execute the script only with Linux and anaconda installed

conda create -n automl_project python=3.7
conda activate automl_project

conda install pandas
conda install -c conda-forge py-xgboost
pip install openml