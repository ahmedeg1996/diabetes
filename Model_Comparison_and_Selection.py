import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

dataset = pd.read_csv('diabetess.csv')

dataset = dataset.values
X = dataset[:,0:8]
y = dataset[:,8]

scalar = StandardScaler().fit(X)
X_scaled = scalar.transform(X)
np.set_printoptions(precision = 3)
models = []

models.append(('LR' , LogisticRegression(solver = 'liblinear')))

models.append(('LDR' , LinearDiscriminantAnalysis()))

results = []
names = []
scoring = 'accuracy'
for name , model in models:

    kfold = KFold(n_splits = 10,random_state = 7 , shuffle = True)
    val_score = cross_val_score(model,X_scaled , y , cv = kfold , scoring = scoring)
    results.append(cross_val_score)
    names.append(name)
    print('the cross_val_score for %s model is %.3f%%' %(name , val_score.mean()*100))
