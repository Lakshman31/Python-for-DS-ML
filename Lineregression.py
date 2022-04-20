import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
df1 = pd.read_csv('covidcases.csv',parse_dates='True')
df = df1.dropna()
app = FastAPI()
#df['category']=pd.to_datetime(df['category'])
X = np.array(df['Actual Data'].values).reshape(-1,1)
Y = np.array(df.index.values).reshape(-1,1)
W = np.array(df['Model Computed Data'].values).reshape(-1,1)
reg=linear_model.LinearRegression()
reg.fit(X,Y)
val = np.array(float(input("enter the day"))).reshape(-1,1)
z=reg.predict(val)
print(z)
r2_score = reg.score(X,Y)
print(r2_score)
