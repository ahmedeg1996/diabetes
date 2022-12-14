import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pickle

dataset = pd.read_csv("diabetess.csv")
dataset = dataset.values
X = dataset[:,:8]
y = dataset[:,8]
scalaing = StandardScaler().fit(X)
X_scaled = scalaing.transform(X)
X_train , X_test , y_train , y_test = train_test_split(X_scaled , y , test_size = 0.3 , random_state = 42)
model = LogisticRegression(solver = 'liblinear')
model.fit(X_train , y_train)
file_name = 'my_logistic_model'
pickle.dump(model,open(file_name , 'wb'))

#loading the model

model = pickle.load(open(file_name , 'rb'))
test = np.array([[-0.54791859,-0.27837344 ,0.304734,0.71908574,-0.69289057,0.47054319,-0.97814487,-1.04154944]])
print(model.predict(test))

