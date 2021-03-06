import pandas as pd
import matplotlib.pyplot as plt

data_original = pd.read_csv('covidcases.csv')
data = data_orginal.dropna()
print(data.isnull().sum())
plt.plot(data['category'],data['Actual Data'],'b',label='Actual data')
plt.plot(data['category'],data['Model Computed Data'],'g',label='Predicted data')
plt.xlabel("Date")
plt.ylabel("Covid cases")
plt.legend()
plt.show()
