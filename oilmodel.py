## Top line code for our Oil Equity Model
## We are using 4 test firms (ExxonMobil, Diamondback Resources, Devon Energy and ConocoPhillps)

# packages needed
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
import os
import json
import requests

# Import Tickers
tickerList = pd.read_csv(
    r"C:\Users\MichaelTanner\Documents\code_doc\oilstock\tickersOilModel.csv"
)

urlone = "https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol="
urltwo = "&apikey=KDQ6SCUZM31MB7PW"  # alpha vantage API key

for i in range(0, 1):
    urlNew = urlone + str(tickerList.at[i, "Ticker"]) + urltwo
    r = requests.get(urlNew)
    print(r.text)


# Create Variance Inflation Factor program
def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif


# Pulls in clean csv (we are lazy coders)
dataOil = pd.read_csv(
    r"C:\Users\MichaelTanner\Documents\code_doc\oilstock\dataOilModel.csv"
)

dataOilModel = dataOil.copy(deep=True)
dataOilModel = dataOilModel.drop("Date", axis=1)

vifTable = dataOilModel.iloc[:, :-1]
vifbetter = calc_vif(vifTable)

print(vifbetter)

# print(dataOil.info())

xvar = dataOil[
    [
        "CPI",
        "Federal Funds Rate",
        "Treasury Yield",
        "Inflation",
        "Consumer Sentiment",
        "Oil Price",
        "US Oil Production",
        "US Oil Consumption",
    ]
]


# ExxonMobil
yvarExxon = dataOil["Exxon"]

xvar = sm.add_constant(xvar)
model = sm.OLS(yvarExxon, xvar.astype(float)).fit()
predications = model.predict(xvar)

printm = model.summary()
print(printm)

# Devon Energy
yvarDevon = dataOil["Devon"]

xvar = sm.add_constant(xvar)
model = sm.OLS(yvarDevon, xvar.astype(float)).fit()
predications = model.predict(xvar)

printm = model.summary()
print(printm)

# ConocoPhillps
yvarCop = dataOil["ConocoPhillips"]

xvar = sm.add_constant(xvar)
model = sm.OLS(yvarCop, xvar.astype(float)).fit()
predications = model.predict(xvar)

printm = model.summary()
print(printm)


print("dataOil")
