import datetime
import numpy as np
from fastapi import FastAPI,Body,Request,Form
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel
from sklearn import linear_model
from statsmodels.tsa.stattools import adfuller
from fastapi.responses import HTMLResponse
from fastapi.responses import ORJSONResponse
# Creating FastAPI instance
app = FastAPI()
templates = Jinja2Templates(directory="htmldir")
# Creating class to define the request body
# and the type hints of each attribute

df1 = pd.read_csv('C:/Users/laksh/Downloads/covidcases.csv')
df = df1.dropna()
#df['category']=pd.to_datetime(df['category'])
print(df.isnull().sum())
#df=df.set_index('category',drop=True)

# Getting our Features and Targets
def ad_Test(dataset):
    dftest=adfuller(dataset,autolag='AIC')
    print(dftest[1])
ad_Test(df['Actual Data'])
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
step_fit = auto_arima(df['Actual Data'],trace=True,supress_warning =True)
from statsmodels.tsa.arima.model import ARIMA
train= df.iloc[:-10]
test = df.iloc[-10:]
model=ARIMA(train['Actual Data'].values,order=(2,1,1))
model=model.fit()
#Predictions
start = len(train)
end = len(train)+len(test)-1
pred = pd.DataFrame(model.predict(start= start,end = end,type='levels'))
pred.index= df.index[start:end+1]

future_dates = pd.date_range(start='2022-03-21',end='2022-05-01')
pred = pd.DataFrame(model.predict(start= len(df),end = len(df)+41,type='levels'))
pred.index= future_dates
#print(pred)
print((pred[0].index[1]).date())
dt=(datetime.datetime.strftime((pred[0].index[1]).date(),"%#m/%#d/%y"))

# Creating an Endpoint to receive the data
# to make prediction on.

@app.get("/home",response_class=HTMLResponse)
def write_test(request:Request,predicted:HTMLResponse):
    return templates.TemplateResponse("test.html",{"request":request,"predicted":predicted})

@app.post("/home")
async def handleform(request:Request,predicted:HTMLResponse,modelpred: float| None = None,err: str| None = None,Date:str = Form(...)):
    try:
        d1=pd.to_datetime(Date).date()
        Date=datetime.datetime.strftime(d1,"%#m/%#d/%y")
        c = 0
        for i in range(444):
            if df1['category'].values[i] in Date:
                predicted = df1['Actual Data'][i]
                modelpred = df1['Model Computed Data'][i]
                #return templates.TemplateResponse("test.html",{"request":request,"date":Date,"predicted":predicted,"predicted1":modelpred})
                c = 1

        if c == 0:
            for j in range(41):
                if datetime.datetime.strftime((pred[0].index[j]).date(), "%#m/%#d/%y") in Date:
                    predicted  = 0
                    modelpred = round(pred[0][j])
                    #return templates.TemplateResponse("test.html",{"request":request,"date": Date,"predicted":predicted,"predicted1":modelpred})
                c = 1
        if c == 0:
            predicted = 0
            modelpred = 0
        err=""
    except:
        err = "Sorry! we are unable to recognize the date"
        Date = ""
    return templates.TemplateResponse("test.html", {"request": request, "date": Date, "predicted": predicted,"predicted1": modelpred,"err":err})
