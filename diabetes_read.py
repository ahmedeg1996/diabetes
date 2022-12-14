import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('diabetess.csv')

values = dataset.values

X = values[:,0:8]
Y = values[:,8]
scalar = StandardScaler().fit(X)
rescaled = scalar.transform(X)

np.set_printoptions(precision = 3)

print(rescaled[0:5,:])