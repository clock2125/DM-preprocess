import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

df_data = pd.read_csv('data/pitches.csv')
# df_data = df_data.drop(['Unnamed: 0'], axis=1)

df_numeric = df_data.select_dtypes(exclude='O')
df_noNumeric = df_data.select_dtypes(include='O')


# 使用最高频率值来填补缺失值
simpleImp = SimpleImputer(strategy="most_frequent")
data_columns = df_data.columns
df_nona = pd.DataFrame(simpleImp.fit_transform(df_data))
df_nona.columns = data_columns

df_nona.to_csv('results/pitches-nona.csv', index=False)

