import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('diabetess.csv')
dataset = dataset.values
X = dataset[:,0:8]
Y = dataset[:,8]
dataset = StandardScaler().fit(X)
dataset = dataset.transform(X)
np.set_printoptions(precision = 3)

kfold = KFold(10,random_state = 7 , shuffle = True)
model = LogisticRegression(solver = 'liblinear')
cross_val_score = cross_val_score(model,X,Y,cv = kfold , scoring = 'neg_log_loss')

print("validation score =%.3f" % (cross_val_score.mean()*100))
