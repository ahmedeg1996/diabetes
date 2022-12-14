import numpy as np
import pandas as  pd

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('diabetess.csv')
dataset = dataset.values
X = dataset[:,0:8]
y = dataset[:,8]

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
parameters = np.array([0.001,0.01,0.1,1])
grid = dict(alpha = parameters)

model = Ridge()

grid_search= GridSearchCV(model ,grid , cv = 3)
grid_search.fit(X_scaled , y)
print(grid_search.best_estimator_.alpha)

