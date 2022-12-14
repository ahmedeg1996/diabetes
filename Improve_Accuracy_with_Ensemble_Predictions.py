import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('diabetess.csv')

values = dataset.values

X = values[:,0:8]
y = values[:,8]
scalar = StandardScaler().fit(X)
rescaled = scalar.transform(X)

np.set_printoptions(precision = 3)

model = RandomForestClassifier(n_estimators=100,max_features = 3)
kfold = KFold(10)
val_score = cross_val_score(model , X , y , cv = kfold)
print("score :%3f" %(val_score.mean()))