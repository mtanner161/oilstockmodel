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

# Import Ticker Data
tickerData = pd.read_csv(
    r"C:\Users\MichaelTanner\Documents\code_doc\oilstock\tickersDataModel.csv"
)

tickerDataNp = tickerData.to_numpy()

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


for i in range(0, len(tickerList)):

    yvar = tickerDataNp[:, i + 1]  ####LEFT OF HERE
    df = pd.DataFrame(yvar)
    xvar = sm.add_constant(xvar)
    model = sm.OLS(df, xvar.astype(float)).fit()
    predications = model.predict(xvar)

    printm = model.summary()
    print(printm)

    r = np.zeros([8, 1], dtype=float)
    r = model.tvalues
    tvalue = r["Oil Price"]

    print(tvalue)
