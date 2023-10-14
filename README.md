# Supervised Learning for Credit Risk Classification
Train and evaluate models to classify loan risks.

## File structure
* The `img` directory contains PNG images of the charts generated in the analysis
* The `Resources` directory contains the lending data provided by Monash University
* The `credit_risk_classification.ipynb` Jupyter notebook contains the main analysis, including the code and results
* The `ml_classification.py` module contains functions that are commonly used in classification projects (supervised learning)
* The `models_optimisation.ipynb` Jupyter notebook contains contains a side analysis with the aim to optimise some of the models' parameters
* The `style.py` module contains variables used to define the style of the charts, such as colors

All code is the author's, unless otherwise specifically specified.

This README files contains the main points from the analysis and our conclusions. The most complete and up-to-date report can be found in the `credit_risk_classification.ipynb` Jupyter notebook itself, from which most of the text below is copied for convenience.

## Overview of the Analysis
### Purpose
In this analysis, we look at the ability of different machine learning (ML) models to classify healthy and high-risk loans.

### About the data
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).


### Methods used
The following models form the `sklearn` library are evaluated:
* `LogisticRegression` with original data (Model 1) and scaled data (Model 2)
* `SVC` with scaled data (Model 3)
* `tree` with scaled data (Model 4)
* `RandomForest` with scaled data (Model 5)
* `KNeighborsClassifier` with scaled data (Model 6)

Note that PCA is not used as the number of features (dimensions) is reasonable and we do not expect any significant improvement by using Principal Components.

### Stages
We prepare the data for all the models in the next sections by performing the following steps:
* Import all necessary modules (there are no imports within the other code blocks)
* Load the data from the CSV file into a pandas DataFrame
* Split the data between the training and test sets using `train_test_split` from `sklearn`

For each of the models, we then perform the following steps
* Scaling (optional)
* Fitting (i.e. train the model with the training set)
* Predictions (i.e. use the test set )
* Describe the stages of the machine learning process you went through as part of this analysis.
