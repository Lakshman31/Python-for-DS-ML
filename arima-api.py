import datetime
import numpy as np
from fastapi import FastAPI,Body,Request,Form
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from fastapi.responses import HTMLResponse
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.arima.model import ARIMA
import lineregress


def arima_api():

    def ad_Test(dataset):
        datatest=adfuller(dataset,autolag='AIC')
    ad_Test(data['Actual Data'])
    warnings.filterwarnings("ignore")
    step_fit = auto_arima(data['Actual Data'],trace=True,supress_warning =True)

    #Splitting the data for testing and training
    train= data.iloc[:-10]
    test = data.iloc[-10:]
    model=ARIMA(train['Actual Data'].values,order=(2,1,1))
    model=model.fit()

    #Predictions
    start = len(train)
    end = len(train)+len(test)-1
    pred = pd.DataFrame(model.predict(start= start,end = end,type='levels'))
    pred.index= data.index[start:end+1]

    future_dates = pd.date_range(start='2022-03-21',end='2022-05-01')
    pred = pd.DataFrame(model.predict(start= len(data),end = len(data)+len(future_dates)-1,type='levels'))
    pred.index= future_dates
    return (pred)

#FastApi
app = FastAPI()
templates = Jinja2Templates(directory="htmldir")
data_original = pd.read_csv('covidcases.csv')
data = data_original.dropna()
print(data.isnull().sum())


@app.get("/home",response_class=HTMLResponse)
def write_test(request:Request,predicted:HTMLResponse):
    return templates.TemplateResponse("fapi.html",{"request":request,"predicted":predicted})

@app.post("/home")
async def handleform(request:Request,predicted:HTMLResponse,modelpred: float| None = None,err: str| None = None,Date:str = Form(...)):
    try:
        date_org=pd.to_datetime(Date).date()
        Date=datetime.datetime.strftime(date_org,"%#m/%#d/%y")
        c = 0
        for i in range(len(data)):
            if data_original['category'].values[i] in Date:
                predicted = data_original['Actual Data'][i]
                modelpred = data_original['Model Computed Data'][i]
                c = 1

        if c == 0:
            pred = arima_api()
            for j in range(len(pred)):
                if datetime.datetime.strftime(pred[0].index[j].date(), "%#m/%#d/%y") in Date:
                    predicted  = 0
                    modelpred = round(pred[0][j])
                    c = 1
        if c == 0:
            predicted = 0
            modelpred = 0
        err=""
    except:
        err = "Sorry! we are unable to recognize the date"
        Date = ""
    return templates.TemplateResponse("fapi.html", {"request": request, "date": Date, "predicted": predicted,"predicted1": modelpred,"err":err})
