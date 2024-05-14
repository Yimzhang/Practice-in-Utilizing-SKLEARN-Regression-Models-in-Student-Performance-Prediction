# Group 4: The IT Crowd
# Yiming Zhang: Data cleaning and feature selection.
# Chenlong Xiao: Model evaluation and analysis goodness of fit.
# Yachun Deng: Create prediction and analysis result.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn: statistical data visualization
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

student_performance = pd.read_csv("student-mat.csv")

print(student_performance.describe())  # we first look at stastic description
grade = student_performance['G3']
sns.set(rc={'figure.figsize': (11, 8)})  # set plot to specific size
sns.distplot(grade, bins=35)
plt.show()  # show students' grade (which is our dependent variable)

# 复制原数据，进行数据清洗
df = student_performance.copy()

# data cleaning
# drop columns which contain more than 70 percent of missing value
numer_cols = df.loc[:, df.dtypes != 'object'].columns
categ_cols = df.loc[:, df.dtypes == 'object'].columns
df = df.dropna(thresh=0.7 * df.shape[0], axis=1)

# replace missing data with mode or column
for col in numer_cols:
    m = round(df[col].mean())
    df[col].fillna(value=m, inplace=True)
for col in categ_cols:
    m = df[col].mode()[0]
    df[col].fillna(value=m, inplace=True)

df2 = pd.get_dummies(df)
# construct new features: G0, which is the weighted average of G1 and G2
df2['G0'] = 0.4 * df2['G1'] + 0.6 * df2['G2']
# construct new features: Pedu, which is the sum of Medu and Fedu, represents for parents' education level
df2['Pedu'] = df2['Medu'] + df2['Fedu']

correlation_matrix = df2.corr(method='pearson').round(2)
correlation_matrix = np.absolute(correlation_matrix)
index = correlation_matrix['G3'].sort_values().index
features = index[::-1]
print(f'list of features in covariance descending order:\n{features}')

# 我们用到的features：我尝试削减featrues到5个以下，但是跑出来r2在0.3左右，也就是说features选取严重不足，很难预测学生成绩
# 所以选了前18个变量来进行建模，其中G0和Pedu是我们自己构建的，其余的带'_'的是dummy化的category data
# 格式为：变量名称_选项，例如'Mjob_health'代表了在Mjob这个变量中结果是health的，值设为1，其他选项（athome，teacher）值为0
features = ['G0', 'failures', 'Pedu', 'higher_yes', 'age', 'romantic_no', 'goout', 'Mjob_at_home',
            'traveltime', 'Mjob_health', 'address_U', 'sex_F', 'reason_course', 'paid_no', 'reason_reputation',
            'studytime', 'internet_no', 'famsize_GT3']

y = df2['G3']
x = df2[features]

# features co-linear check
correlation_matrix = x.corr(method='pearson').round(2)  # 2 decimal accuracy
sns.heatmap(data=correlation_matrix, annot=True)
plt.title('Pearson pair-wise Correlation Matrix the features')
plt.show()

'''evaluate regression result and model'''


def adj_r2(n, p, r2):  # define adjusted r2
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# set training size as 0.8, test size as 0.2 of entire training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99)
# (evaluation) run multiple regression on training set
n_train, p_train = x_train.shape[0], x_train.shape[1]
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_train_predict = lr_model.predict(x_train)
rmse_train = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2_train = r2_score(y_train, y_train_predict)
a_r2_train = adj_r2(n_train, p_train, r2_train)
print("Multiple Lin. Regression Model performance on training set:")
print(f'RMSE = {rmse_train:0.4f}')
print(f'R^2 = {r2_train:0.4f}')
print(f'Adj R^2 = {a_r2_train:0.4f}')
print('*' * 50)

# (evaluation) run multiple regression on testing set
n_test, p_test = x_test.shape[0], x_test.shape[1]
y_test_predict = lr_model.predict(x_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)
a_r2_test = adj_r2(n_test, p_test, r2_test)
print("Multiple Lin. Regression Model performance on testing set:")
print(f'RMSE = {rmse_test:0.4f}')
print(f'R^2 = {r2_test:0.4f}')
print(f'Adj R^2 = {a_r2_test:0.4f}')
print('*' * 50)

# print the coefficient of features and G3
a = features.copy()
coefficient_table: DataFrame = pd.DataFrame(np.nan, index=a, columns=['G3'])
coefficient_table['G3']= lr_model.coef_
print(f'coefficient table: \n{coefficient_table}')
print('*' * 50)

# plot regression result
y=y_test
y=y.to_numpy()
plt.scatter(y,y_test_predict)
plt.plot(y,y,'r-')
plt.title('Linear Regression Evaluation')
plt.xlabel('Real G3')
plt.ylabel('Predicted G3')
plt.savefig('Linear Regression Evaluation')
plt.show()

# (evaluation) run polynominal regression on training set
poly_features = PolynomialFeatures(degree=2)
x_train_poly = poly_features.fit_transform(x_train)
lr_model_poly = LinearRegression()
lr_model_poly.fit(x_train_poly, y_train)
y_train_predict_poly = lr_model_poly.predict(x_train_poly)
rmse_train_poly = np.sqrt(mean_squared_error(y_train, y_train_predict_poly))
r2_train_poly = r2_score(y_train, y_train_predict_poly)
print("Polynomial Regression Model performance on training set:")
print(f'RMSE = {rmse_train_poly:0.4f}')
print(f'R^2 = {r2_train_poly:0.4f}')
print('*' * 50)

# (evaluation) run polynominal regression on testing set
x_test_poly = poly_features.fit_transform(x_test)
y_test_predict_poly = lr_model_poly.predict(x_test_poly)
rmse_test_poly = (np.sqrt(mean_squared_error(y_test, y_test_predict_poly)))
r2_test_poly = r2_score(y_test, y_test_predict_poly)
print("Polynomial Regression Model performance on testing set:")
print(f'RMSE = {rmse_test_poly:0.4f}')
print(f'R^2 = {r2_test_poly:0.4f}')
print('*' * 50)

plt.scatter(y,y_test_predict_poly)
plt.plot(y,y,'r-')
plt.title('Polynominal Regression Evaluation')
plt.xlim(-1,4+max(y.max(),y_test_predict_poly.max()))
plt.ylim(-1,4+max(y.max(),y_test_predict_poly.max()))
plt.xlabel('Real G3')
plt.ylabel('Predicted G3')
plt.savefig('Polynominal Regression Evaluation')
plt.show()

# # run linear regression on future data (for future prediction)
# lr_model = LinearRegression()
# lr_model.fit(x, y)
# n = x.shape[0]
# p = x.shape[1]
# y_predict_linear = lr_model.predict(x)
# rmse = (np.sqrt(mean_squared_error(y, y_predict_linear)))
# r2 = r2_score(y, y_predict_linear)
# a_r2 = adj_r2(n, p, r2)
# print("Multiple Lin. Regression Model performance:")
# print(f'R^2 = {r2:0.4f}')
# print(f'RMSE = {rmse:0.4f}')
# print(f'Adj R^2 = {a_r2:0.4f}')
