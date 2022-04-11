import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
df1 = pd.read_csv('covidcases.csv')
df = df1.dropna()
print(df.isnull().sum())
reg = linear_model.LinearRegression()
X = np.array(df["Actual Data"].values).reshape(-1,1)
Y = np.array(df["Model Computed Data"].values).reshape(-1,1)
reg.fit(X,Y)
Z = reg.predict(X)
r2_score = reg.score(X,Y)
print(r2_score)
plt.scatter(X,Y,color="Green",label="Data collected")
plt.plot(X,Z,color = "Blue",label="regression line")
plt.legend()
plt.show()
