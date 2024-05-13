# Vehicle Insurance Claims Prediction

## Overview
This project aims to predict vehicle insurance claims using machine learning techniques. It involves data preprocessing, feature engineering, model training, and evaluation.

## Libraries Used
- numpy
- pandas
- matplotlib
- seaborn
- gensim
- joblib
- xgboost
- hyperopt
- sklearn

## Project Structure
- **Claims_prediction.ipynb**: Jupyter notebook containing all the code for data preprocessing, feature engineering, model training, and evaluation.
- **README.md**: Overview of the project, instructions for setup, and usage guidelines.

## Setup
1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Open and run `Claims_prediction.ipynb` in Jupyter Notebook or JupyterLab.

## Usage
1. Follow the instructions in `Claims_prediction.ipynb` to explore the data, preprocess it, engineer features, train machine learning models, and evaluate their performance.
2. Ensure that the necessary dataset files (`train_set.csv`, `test_set.csv`, `example_entry.csv`) are in the parent directory.

## Files Description
- **Claims_prediction.ipynb**: Jupyter notebook containing all the code for data preprocessing, feature engineering, model training, and evaluation.
- **train_set.csv**: Training dataset containing features and target variable (`Claim_Amount`).
- **test_set.csv**: Test dataset for prediction.
- **example_entry.csv**: Example entry file for merging with the test dataset.
- **imp_feats_volatile_sampling.csv**: Feature importance data from volatile sampling method.
- **imp_feats_fixed_sampling.csv**: Feature importance data from fixed sampling method.
- **lr_model_5.pkl**: Trained Linear Regression model.
- **xgb_model_5.pkl**: Trained XGBoost model.
- **xgb_model_hyperopt_10.pkl**: Tuned XGBoost model using hyperparameter optimization.

## Results
- **Linear Regression Model**:
  - Root Mean Squared Error: [RMSE]
  - Normalized Gini Coefficient: [Gini Coefficient]
- **XGBoost Model** (Base):
  - Root Mean Squared Error: [RMSE]
  - Normalized Gini Coefficient: [Gini Coefficient]
- **XGBoost Model** (Tuned with Hyperopt):
  - Root Mean Squared Error: [RMSE]
  - Normalized Gini Coefficient: [Gini Coefficient]

## Author
Anoop CA