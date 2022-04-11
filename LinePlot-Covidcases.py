import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv('C:/Users/laksh/Downloads/covidcases.csv')
df = df1.dropna()
print(df.isnull().sum())
plt.plot(df['category'],df['Actual Data'],'b',label='Actual data')
plt.plot(df['category'],df['Model Computed Data'],'g',label='Predicted data')
plt.xlabel("Date")
plt.ylabel("Covid cases")
plt.legend()
plt.show()
