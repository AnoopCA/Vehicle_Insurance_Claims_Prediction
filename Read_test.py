import numpy as np
import pandas as pd

data = pd.read_csv('train_set.csv')
#print(data.head())
#print(data.columns)
#print(data['Household_ID'].head(50))
print(data['Claim_Amount'].sort_values(ascending=False).head())
