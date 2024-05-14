import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn: statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# part a
data = pd.read_csv("数据源.csv")  # read in csv data as dataframe(subdirectory/file)

# len(ames_train.columns) #there are 82 columns(features)

correlation_matrix = data.corr(method='pearson').round(2)
correlation_matrix = np.absolute(correlation_matrix)
index = correlation_matrix['G3'].sort_values().index
features = index[::-1]
print(features)
print(correlation_matrix)




# part b
price = ames_train['SalePrice']
sns.set(rc={'figure.figsize': (11, 8)})  # set plot to specific size
sns.distplot(price, bins=35)
plt.show()
# Yes, the histgram is right skewed.

# part c
ames_train['SalePrice'] = np.log(ames_train['SalePrice'])
price = ames_train['SalePrice']
sns.set(rc={'figure.figsize': (11, 8)})  # set plot to specific size
sns.distplot(price, bins=35)
plt.show()

# part d

# use .dropna method using axis ="column" to drop column
# drop column that have more than 70% missing values
# aka drop columns that have #na's>1022
ames_train = ames_train.dropna(thresh=0.7 * ames_train.shape[0], axis=1)

# part e
numer_cols = ames_train.loc[:, ames_train.dtypes != 'object'].columns
categ_cols = ames_train.loc[:, ames_train.dtypes == 'object'].columns
for col in numer_cols:
    m = round(ames_train[col].mean())
    ames_train[col].fillna(value=m, inplace=True)
for col in categ_cols:
    m = ames_train[col].mode()[0]
    ames_train[col].fillna(value=m, inplace=True)

# part f
new_numer_cols = ames_train.loc[:, ames_train.dtypes != 'object']

# part g
