# kuebiko - mansa data coding challenge - Oscar Gilg

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This folder contains a solution to the Mansa data science challenge.

### Notebooks

- EDA: Exploratory data analysis, plotting data, preprocessing, encoding parts of the data, feature extraction/engineering.
- ModelBuilding: Implemented and tested 3 models: Linear Regression, XGBoost and Neural Network to predict expense given 3 months of past data for a user. 

### Python files

- main.py: Integrating model with FastAPI for prediction on request
- test_main.py: Testing the FastAPI prediction with a dummy account.

### Folders

- src: contains classes and functions for feature preprocessing and feature engineering.
- pickle: containes the model, encoders and feature list as pickle files
- data: contains data in csv format

