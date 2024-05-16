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
- **Data/**: Contains the dataset files (`train_set.csv`, `test_set.csv`, `example_entry.csv`).
- **Models/**: Stores trained models and feature importance data.
- **Notebooks/**: Jupyter notebooks for exploratory data analysis, feature engineering, and model training.
- **readme.md**: Overview of the project, instructions for setup, and usage guidelines.

## Setup
1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Place the dataset files (`train_set.csv`, `test_set.csv`, `example_entry.csv`) in the `Data/` directory.
3. The size of the data is heavy and please download it from the actual Kaggle competition. Link: https://www.kaggle.com/c/ClaimPredictionChallenge/data

## Usage
1. Explore the data using Jupyter notebooks in the `Notebooks/` directory.
2. Preprocess the data and engineer features using scripts in the `Scripts/` directory.
3. Train machine learning models using the provided scripts.
4. Evaluate model performance and generate predictions.

## Files Description
- **train_set.csv**: Training dataset containing features and target variable (`Claim_Amount`).
- **test_set.csv**: Test dataset for prediction.
- **example_entry.csv**: Example entry file for merging with the test dataset.
- **imp_feats_volatile_sampling.csv**: Feature importance data from volatile sampling method.
- **imp_feats_fixed_sampling.csv**: Feature importance data from fixed sampling method.
- **lr_model_5.pkl**: Trained Linear Regression model.
- **xgb_model_5.pkl**: Trained XGBoost model.
- **xgb_model_hyperopt_10.pkl**: Tuned XGBoost model using hyperparameter optimization.

## Scripts
1. `Claims_Prediction-FE.py`: Performs feature engineering techniques such as creating new features and word vectorization.

## Notebooks
1. `exploratory_data_analysis.ipynb`: Notebook for exploring the dataset, visualizing distributions, and understanding feature relationships.
3. `model_training_evaluation.ipynb`: Notebook for training machine learning models, evaluating performance, and generating predictions.

## Results
- **Linear Regression Model**:
  - Mean Squared Error: [MSE]
  - Root Mean Squared Error: [RMSE]
  - Normalized Gini Coefficient: [Gini Coefficient]
- **XGBoost Model** (Base):
  - Mean Squared Error: [MSE]
  - Root Mean Squared Error: [RMSE]
  - Normalized Gini Coefficient: [Gini Coefficient]
- **XGBoost Model** (Tuned with Hyperopt):
  - Mean Squared Error: [MSE]
  - Root Mean Squared Error: [RMSE]
  - Normalized Gini Coefficient: [Gini Coefficient]

## Author
Anoop CA