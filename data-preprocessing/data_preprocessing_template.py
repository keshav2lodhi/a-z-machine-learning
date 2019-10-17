# Data Pre-processing

"""Importing the Libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Importing the dataset"""
data_set = pd.read_csv('/Users/keshavlodhi/PycharmProjects/a-z-machine-learning/data-sets/Data.csv')
# iloc contains the lines(rows) before the ',' and columns after ','. Here -1 means excluding the last column
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 3].values

"""Taking care of missing data"""
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

"""Encoding categorical data"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoding for country column
label_encoder_x = LabelEncoder()
x[:, 0] = label_encoder_x.fit_transform(x[:, 0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
x = one_hot_encoder.fit_transform(x).toarray()
# Encoding for purchased column
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
print(x)
print(y)
