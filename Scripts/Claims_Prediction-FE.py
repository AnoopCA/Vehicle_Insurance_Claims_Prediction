# Vehicle Insurance Claims Prediction

# The "Vehicle Insurance Claims Prediction" project was part of the "Allstate Claim Prediction Challenge" on Kaggle, held on 13-07-2011,
# with a focus on predicting claims payments. The goal was to develop predictive models to estimate insurance claims payments based on historical data
# and relevant features, using various machine learning techniques, data preprocessing methods, and feature engineering approaches to improve prediction accuracy.

# Import the necessary libraries:
import numpy as np
import pandas as pd
# Import KeyedVectors from gensim.models for working with word vectors
from gensim.models import KeyedVectors
# Import StandardScaler from sklearn.preprocessing for standardization
from sklearn.preprocessing import StandardScaler

# Import train and test datasets
x_train = pd.read_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\Data\train_set.csv')

x_test = pd.read_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\Data\test_set.csv')
example_entry = pd.read_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\Data\example_entry.csv')
x_test = x_test.merge(example_entry, on='Row_ID', how='left')

# Due to the large size of the dataset, training the model with the full dataset is difficult, so we sample 60% of the data for training.
#x_train = x_train.sample(frac=0.60)
#x_test = x_test.sample(frac=0.60)

#x_train.to_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\x_train_60.csv', index=False)
#x_test.to_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\x_test_60.csv', index=False)

#x_train = pd.read_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\x_train_60.csv')
#x_train = x_train.sample(frac=1)
#x_test = pd.read_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\x_test_60.csv')

# Data Preprocessing:

# Preprocess the data by removing rows with Claim_Amount > 100, handling "?" values in categorical features, and filling missing values in 'Cat12'.

# Remove rows from x_train where Claim_Amount is greater than 100
x_train = x_train.drop(x_train[x_train['Claim_Amount']>100].index, axis=0)
# Remove rows from x_test where Claim_Amount is greater than 100
x_test = x_test.drop(x_test[x_test['Claim_Amount']>100].index, axis=0)

# Identify categorical features with "?" values in x_train
cat_feats = x_train.columns[x_train.apply(lambda x: x == "?").any()]

# Initialize a list to store data about "?" occurrences
q_data = []
# Iterate over categorical features with "?" values
for i in cat_feats:
    # Append the feature name, count "?" occurrences and the corresponding percentages to the list "q_data"
    q_data.append([i, len(x_train[x_train[i]=="?"][i]), round((len(x_train[x_train[i]=="?"][i]) / len(x_train))*100, 4)])

# Create a DataFrame from the "?" data list created
q_data = pd.DataFrame(q_data, columns=['Feature name', 'No of "?"', 'Percentage of Occurrence'])
# Filter features with more than 100000 "?" occurrences
q_cols = q_data[q_data['No of "?"'] > 100000]['Feature name']

# Handle specific cases in x_train for certain features having "?" in any of the categorical features.
for i in cat_feats:
    if i == "OrdCat":
        # Convert "OrdCat" to numeric and fill missing values
        x_train[i] = pd.to_numeric(x_train[i], errors='coerce')
        x_train[i] = x_train[i].fillna(x_train.loc[x_train[i].notna(), i].mode()[0])
        x_train[i] = x_train[i].astype(int)
    elif i in q_cols.values:
        # Replace "?" with the most frequent non-"?" value for selected features
        x_train[i] = x_train[i].replace("?", x_train[x_train[i]!="?"][i].value_counts().index[0])
    else:
        # Replace "?" with "Z" for other features
        x_train[i] = x_train[i].replace("?", "Z")

# Fill missing values in 'Cat12' with "Z"
x_train['Cat12'].fillna('Z', inplace=True)

# Feature Engineering:

# This section performs extensive feature engineering on the input DataFrame, including numeric calculations, logarithmic and
# exponential transformations, categorical feature grouping, word vector generation, and feature importance-based feature selection.
# Additionally, it scales the train and test datasets using "StandardScaler" for normalization, preparing the data for subsequent modeling and evaluation.

def Feature_Engineering(data):
    """
    Perform feature engineering on the input DataFrame.

    Parameters:
    data (DataFrame): Input DataFrame containing relevant columns.

    Returns:
    DataFrame: DataFrame with engineered features.
    """
    # Calculate numeric differences and operations between Calendar Year and Model Year
    data['Calendar_Year_Nums'] = data['Calendar_Year']-2000  # Calculate years since 2000
    data['Model_Year_Nums'] = data['Model_Year']-1980  # Calculate years since 1980
    data['Calendar_Year_Model_Year_Add'] = data['Model_Year_Nums'] + data['Calendar_Year_Nums']  # Add both year counts
    data['Calendar_Year_Model_Year_Sub'] = data['Model_Year_Nums'] - data['Calendar_Year_Nums']  # Subtract year counts
    data['Calendar_Year_Model_Year_Mul'] = data['Model_Year_Nums'] * data['Calendar_Year_Nums']  # Multiply year counts
    data['Calendar_Year_Model_Year_Div'] = data['Model_Year_Nums'] / data['Calendar_Year_Nums']  # Divide year counts

    # Calculate logarithm of Model Year, Calendar Year, and Vehicle columns
    data['Model_Year_Log'] = np.log(data['Model_Year'])  # Logarithm of Model Year
    data['Model_Year_Nums_Log'] = np.log(data['Model_Year_Nums'])  # Logarithm of Model Year Numbers
    data['Calendar_Year_Log'] = np.log(data['Calendar_Year'])  # Logarithm of Calendar Year
    data['Calendar_Year_Nums_Log'] = np.log(data['Calendar_Year_Nums'])  # Logarithm of Calendar Year Numbers
    data['Vehicle_Log'] = np.log(data['Vehicle'])  # Logarithm of Vehicle column

    data['Model_Year_Nums_Exp'] = np.exp(data['Model_Year_Nums'])  # Exponential of Model Year Numbers
    data['Calendar_Year_Nums_Exp'] = np.exp(data['Calendar_Year_Nums'])  # Exponential of Calendar Year Numbers
    data['Vehicle_Exp'] = np.exp(data['Vehicle'])  # Exponential of Vehicle column

    # Replace '?' in 'Blind_Model' column with 'nan.nan'
    data.loc[data['Blind_Model']=="?", 'Blind_Model'] = "nan.nan"
    # Extract characters before '.' in 'Blind_Model' and handle 'nan' values
    data['Blind_Model_Char'] = data['Blind_Model'].str.split(".").str[0]
    data.loc[data['Blind_Model_Char']=='nan', 'Blind_Model_Char'] = data['Blind_Model_Char'].mode()[0]
    # Extract numeric part after '.' in 'Blind_Model' and convert to numeric type
    data['Blind_Model_Num'] = data['Blind_Model'].str.split(".").str[1].str.strip("'")
    data['Blind_Model_Num'] = pd.to_numeric(data['Blind_Model_Num'], errors='coerce')
    data['Blind_Model_Num'].fillna(data['Blind_Model_Num'].mode()[0], inplace=True)
    data['Blind_Model_Num'] = data['Blind_Model_Num'].fillna(0).astype(int)
    # Replace 'nan.nan' in 'Blind_Model' with mode of 'Blind_Model'
    data.loc[data['Blind_Model']=="nan.nan", 'Blind_Model'] = data['Blind_Model'].mode()[0]

    # Replace '?' in 'Blind_Submodel' column with 'nan.nan'
    data.loc[data['Blind_Submodel']=="?", 'Blind_Submodel'] = "nan.nan"
    # Extract numeric part after second '.' in 'Blind_Submodel' and convert to numeric type
    data['Blind_Submodel_Num'] = data['Blind_Submodel'].str.split(".").str[2].str.strip("'")
    data['Blind_Submodel_Num'] = pd.to_numeric(data['Blind_Submodel_Num'], errors='coerce')
    data['Blind_Submodel_Num'].fillna(data['Blind_Submodel_Num'].mode()[0], inplace=True)
    data['Blind_Submodel_Num'] = data['Blind_Submodel_Num'].fillna(0).astype(int)
    # Replace 'nan.nan' in 'Blind_Submodel' with mode of 'Blind_Submodel'
    data.loc[data['Blind_Submodel']=="nan.nan", 'Blind_Submodel'] = data['Blind_Submodel'].mode()[0]

    # Calculate the sum of 'Blind_Model_Num' and 'Blind_Submodel_Num' as 'Model_Nums'
    data['Model_Nums'] = data['Blind_Model_Num'] + data['Blind_Submodel_Num']

    # Do various transformations for columns containing "NVVar" and having length 6
    # Limiting the column name length to 6 to avoid creating features based on the newly created features
    NVV_pairs = []  # List to store unique pairs of NVVar columns
    NVV_cols = [k for k in data.columns if "NVVar" in k and len(k)==6]  # Filter NVVar columns
    for i in NVV_cols:
        # Perform inverse, square root, logarithm, and exponential operations on NVVar columns
        data[i + "_Inv"] = 1 / data[i]
        data[i + "_Sqrt"] = np.sqrt(abs(data[i]))
        data[i + "_Log"] = np.log(abs(data[i]))
        data[i + "_Exp"] = np.exp(data[i])
        # The inner loops and conditions are for avoding addition and multiplication twice on same pair of features
        for j in NVV_cols:
            if ("NVVar" in i) and ("NVVar" in j):
                if i != j:
                    # Calculate differences, ratios, sums, and products for NVVar column pairs
                    data[i + "_" + j + "_Sub"] = data[i] - data[j]
                    data[i + "_" + j + "_Div"] = data[i] / data[j]
                    if ((i, j) not in NVV_pairs) and ((j, i) not in NVV_pairs):
                        NVV_pairs.append((i,j))
                        data[i + "_" + j + "_Add"] = data[i] + data[j]
                        data[i + "_" + j + "_Mul"] = data[i] * data[j]
    # Process "Var" columns
    var_cols = [i for i in data.columns if i.startswith("Var") and len(i)==4]
    # Perform inverse, square root, logarithm, and exponential operations on Var columns
    for i in var_cols:
        data[i + "_Inv"] = 1 / (data[i] + 3)
        data[i + "_Sqrt"] = np.sqrt(data[i] + 3)
        data[i + "_Log"] = np.log(data[i] + 3)
        data[i + "_Exp"] = np.exp(data[i] + 3)

    # Combine categorical columns into groups
    data['Cat_1-4'] = data['Cat1'] + data['Cat2'] + data['Cat3'] + data['Cat4']
    data['Cat_5-8'] = data['Cat5'] + data['Cat6'] + data['Cat7'] + data['Cat8']
    data['Cat_9-12'] = data['Cat9'] + data['Cat10'] + data['Cat11'] + data['Cat12']

    # Load pre-trained Word2Vec model
    word2vec_model = KeyedVectors.load_word2vec_format(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\GoogleNews-vectors-negative300.bin', binary=True)

    # Define function to get word vectors
    def get_word_vector(word):
        try:
            return word2vec_model[word][:5]  # Extract first 5 dimensions of the word vector
        except KeyError:
            return np.zeros(5)  # Return zero vector if word is not found in the Word2Vec model

    # Define categorical columns list
    Cat_list = ['Cat_1-4', 'Cat_5-8', 'Cat_9-12']

    # Iterate through categorical columns and generate word vectors
    for i in Cat_list:
        word_vector_cols = [i + f'_w2v_{n}' for n in range(1, 6)]  # Define new columns for word vectors
        # Apply get_word_vector function to each word in the categorical column and create new columns
        data[word_vector_cols] = pd.DataFrame(data[i].apply(lambda word: get_word_vector(word)).tolist(), index=data.index)
    
    # Get the least important features from the feature importance data and remove them from the train and test data
    imp_feats_vs = pd.read_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\imp_feats_volatile_sampling.csv')
    imp_feats_fs = pd.read_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\models\imp_feats_fixed_sampling.csv')
    # Group by Feature and sum Importance, then sort
    grouped_vs = imp_feats_vs.groupby('Feature')['Importance'].sum().nsmallest(58)
    grouped_fs = imp_feats_fs.groupby('Feature')['Importance'].sum().nsmallest(58)
    # Get common features
    common_features = grouped_vs.index.intersection(grouped_fs.index)
    data.drop(common_features, inplace=True, axis=1)

    # Identify categorical features in the DataFrame
    cat_features = data[data.dtypes[data.dtypes == 'object'].index].columns
    # Iterate through each categorical feature and map values to their normalized counts
    for i in cat_features:
        # Calculate the normalized value counts for the current categorical feature
        x = data[i].value_counts(normalize=True)
        # Map the values in the categorical feature to their normalized counts
        data[i] = data[i].map(x)
    # Return the featurized data
    return data

# Apply feature engineering to the training data
x_train = Feature_Engineering(x_train)
# Apply the same feature engineering to the testing data
x_test = Feature_Engineering(x_test)

# Apply feature engineering to the training data
x_train = Feature_Engineering(x_train)
# Apply the same feature engineering to the testing data
x_test = Feature_Engineering(x_test)

# Check if featurization resulted in any null values in the train and test datasets.
# Calculate the number of missing values in each column of the training data
# train_na_counts = x_train.isnull().sum()
# Sort the columns based on the count of missing values in descending order
# sorted_train_na_counts = train_na_counts.sort_values(ascending=False)

# Calculate the number of missing values in each column of the training data
# test_na_counts = x_test.isnull().sum()
# Sort the columns based on the count of missing values in descending order
# sorted_test_na_counts = test_na_counts.sort_values(ascending=False)

# Based on the previous analysis, the newly created feature "Var1_Sqrt" contains 286 null values. Since there are very few null values, we can impute them with 0.
x_test['Var1_Sqrt'].fillna(0, inplace=True)

# Scale the train and test datasets using "StandardScaler" for normalization.
# Initialize a StandardScaler object for feature scaling
scaler = StandardScaler()

# Fit and transform the training data, excluding 'Row_ID' and 'Claim_Amount' columns
train_scaled = scaler.fit_transform(x_train.drop(['Row_ID', 'Claim_Amount'], axis=1))
# Transform the testing data using the same scaler, excluding 'Row_ID' and 'Claim_Amount' columns
test_scaled = scaler.transform(x_test.drop(['Row_ID', 'Claim_Amount'], axis=1))

# Prepare target variable and adjust a single value in the test target variable for evaluation purposes.
# Extract the target variable 'Claim_Amount' for training and testing data
y_train = x_train['Claim_Amount']
y_test = x_test['Claim_Amount']

# Adjust a single value in the test target variable for evaluation purposes
y_test.iloc[int(len(y_test)/5)] = 0.00001

train_scaled.to_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\Data\train_scaled.csv')
test_scaled.to_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\Data\test_scaled.csv')

y_train.to_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\Data\y_train.csv')
y_test.to_csv(r'D:\ML_Projects\Vehicle_Insurance_Claims_Prediction\Data\y_test.csv')