import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('diabetess.csv')

dataset = dataset.values

X = dataset[:,0:8]
Y = dataset[:,8]

kfold = KFold(10,random_state = 7 , shuffle=True)
model = LogisticRegression(solver ='liblinear')
results = cross_val_score(model,X,Y,cv = kfold)
print(results)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
