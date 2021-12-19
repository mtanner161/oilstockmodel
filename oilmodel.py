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
import matplotlib.pyplot as plt
import openpyxl


# Import Tickers
tickerList = pd.read_csv(
    r"C:\Users\MichaelTanner\Documents\code_doc\oilstock\tickersOilModel.csv"
)

# Import Ticker Data
tickerData = pd.read_csv(
    r"C:\Users\MichaelTanner\Documents\code_doc\oilstock\tickersDataModel.csv"
)

# Create Variance Inflation Factor program
def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif


# function to help with handling NaN values
def numberOfNonNans(data):
    count = 0
    for i in data:
        if not np.isnan(i):
            count += 1
    return count


# Pulls in clean csv (we are lazy coders)
dataOil = pd.read_csv(
    r"C:\Users\MichaelTanner\Documents\code_doc\oilstock\dataOilModel.csv"
)

# copies for transformation
dataOilModel = dataOil.copy(deep=True)
dataOilModel = dataOilModel.drop("Date", axis=1)
# prints VIF table for orginal test
vifTable = dataOilModel.iloc[:, :-1]
vifbetter = calc_vif(vifTable)

# creating xvariable for our OLS model
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

# creating a new table to hold the tvalues from each of the OLS runs
tvalueList = np.zeros([len(tickerList)], dtype=float)

# Loop that runs an OLS for all tickers and extracts tvalue
for i in range(0, len(tickerList)):
    # setting the yvar equal to the correct stock dataColumn
    yvar = tickerData.iloc[:, i + 1]
    # checking to make sure it has enough data (statsmodel needs 20 datapoints to run)
    if numberOfNonNans(yvar) > 20:
        xvar = sm.add_constant(xvar)  # adding a constant
        model = sm.OLS(
            yvar, xvar.astype(float), missing="drop"
        ).fit()  # running the OLS while omitting NaN values
        predications = model.predict(xvar)
        # grabbing the tvalues from the OLS model
        r = np.zeros([8, 1], dtype=float)
        r = model.tvalues
        tvalue = r["Oil Price"]  # getting the specific variable in question
        tvalueList[i] = tvalue  # adding tavlue to master list
    else:  # skips if less than 20 data points
        continue

# adds the ticker smybol with the tvalue
tickerList["T-Values"] = tvalueList.tolist()
tickerList = tickerList.sort_values(
    by=["T-Values"], ascending=False
)  # sorting by largerst to smallest

# exporting to excel for analysis
tickerList.to_excel(
    r"C:\Users\MichaelTanner\Documents\code_doc\oilstock\oilstockmodel\tickertValue.xlsx",
    sheet_name="Data",
)

print("done")
