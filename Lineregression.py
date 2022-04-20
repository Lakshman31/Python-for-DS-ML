import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

data_orginal = pd.read_csv('covidcases.csv',parse_dates='True')
data = data_org.dropna()
X = np.array(data['Actual Data'].values).reshape(-1,1)
Y = np.array(data.index.values).reshape(-1,1)
reg=linear_model.LinearRegression()
reg.fit(X,Y)
val = np.array(float(input("enter the day"))).reshape(-1,1)
z=reg.predict(val)
print(z)
r2_score = reg.score(X,Y)
print(r2_score)
