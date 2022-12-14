import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('housing.csv' ,delim_whitespace = True, names = names)

dataset = dataset.values

X = dataset[:,0:13]
y = dataset[:,13]

scaling = StandardScaler().fit(X)
X_scaled = scaling.transform(X)

kfold = KFold(10,random_state = 7 , shuffle = True)

model = DecisionTreeRegressor()

score = 'neg_mean_squared_error'

cross_val_score = cross_val_score(model,X,y,cv = kfold , scoring = score)

print('score = %.3f' % (cross_val_score.mean()))
