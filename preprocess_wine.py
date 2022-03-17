import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

df_data = pd.read_csv('data/winemag-data-130k-v2.csv')
# df_data = df_data.drop(['Unnamed: 0'], axis=1)

df_numeric = df_data.select_dtypes(exclude='O')
df_noNumeric = df_data.select_dtypes(include='O')

# # 将缺失部分剔除
# df_numeric_nona1 = df_numeric.dropna(axis=0)
# print(df_numeric_nona1.head(10))
# print('---------------------')
# for label, content in df_numeric_nona1.items():
#     df_numeric_nona1.plot.box(column=label)

# # 使用最高频率值来填补缺失值
# simpleImp = SimpleImputer(strategy="most_frequent")
# data_columns = df_numeric.columns
# df_numeric_nona2 = pd.DataFrame(simpleImp.fit_transform(df_numeric))
# df_numeric_nona2.columns = data_columns
# print(df_numeric_nona2.head(10))
# print('---------------------')
# for label, content in df_numeric_nona2.items():
#     df_numeric_nona2.plot.box(column=label)

# 通过属性的相关关系来填补缺失值
# df_numeric_nona3 = df_numeric.copy()
# print(df_numeric_nona3)
# print('--------------')
# df_corr = df_numeric.corr(method='pearson')
# means = df_numeric.mean(axis=0)
# print(df_corr)
# print('----------------')
# print(means)
# print('------------------')
# for label1 in df_numeric.columns:
#     max_corr = 0
#     relate_column = label1
#     for label2 in df_numeric.columns:
#         if label1 == label2:
#             continue
#         else:
#             if df_corr.at[label1, label2] > max_corr:
#                 max_corr = df_corr.at[label1, label2]
#                 relate_column = label2
#     if relate_column != label1:
#         rate = means.at[label1]/means.at[relate_column]
#         df_numeric_nona3[label1].fillna(df_numeric[relate_column].mul(rate), inplace=True)
# print(df_numeric_nona3.head(10))

# 通过数据对象之间的相似性来填补缺失值
# knnImp = KNNImputer(n_neighbors=10)
# data_columns = df_numeric.columns
# df_numeric_nona3 = pd.DataFrame(knnImp.fit_transform(df_numeric))
# df_numeric_nona3.columns = data_columns
# print(df_numeric_nona3.head(10))

# 使用最高频率值来填补缺失值
simpleImp = SimpleImputer(strategy="most_frequent")
data_columns = df_data.columns
df_nona = pd.DataFrame(simpleImp.fit_transform(df_data))
df_nona.columns = data_columns

df_nona.to_csv('results/winemag-data-130k-v2-nona.csv', index=False)

