import requests
import pandas as pd
import numpy as np

print("-- Start Data Download ")
API_KEY = "5Z6D01U9OW2CZ202"
STOCKS = ["ABEA.BE","BMW.DE","BAS.DE",'AAPL','AIR.DE','TSLA',"BAYN.DE",'TSS',"ALV.DE","VIV",'005930.KS']
stock_names = {"ABEA.BE":"Google","TSS":"Total","AIR.DE":"Airbus","BMW.DE":"BMW","BAS.DE":"BASF","AAPL":"Apple","TSLA":"Tesla","BAYN.DE":"Bayer","ALV.DE":"Allianz","VIV":"Telefonica","005930.KS":"Samsung"}


for id in STOCKS:
    company = stock_names[id]
    params = {'function': 'TIME_SERIES_DAILY', 'symbol': id,'outputsize':'full','datatype':'json','apikey':API_KEY}
    raw_data = requests.get("https://www.alphavantage.co/query",params=params)
    print("--- Get Data from ", raw_data.url)

    # Extract position of relevant data in json
    tz = raw_data.text.find("Time Zone")
    start = raw_data.text.find("Time Series (Daily)")+len("Time Series (Daily) :")
    end = len(raw_data.text)-1

    #if id == "BMW.DE":
    #    print(raw_data.text)

    # Subtract Timezone
    timeZone = raw_data.text[tz+len("Time Zone:")+3:start-len("Time Series (Daily) :")-14]
    print("--- Timezone ", timeZone)

    df = pd.read_json(raw_data.text[start:end],orient='index')
    df['Timestamp'] = df.index
    df = df.reset_index()
    df = df[['Timestamp','4. close']]
    df.columns = ['Timestamp',company]

    # Forward Filling
    print("--- Forward Filling ", company)

    df[company].iloc[np.where(df[company] == 0)] = np.nan
    df[company].fillna(method='ffill',inplace=True)

    df[company] = np.log(df[company]) - np.log(df[company].shift(1))
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Save to CSV
    PATH = "data/"+ company + ".csv"
    print("--- Save Dataframe to csv ", PATH )
    df.to_csv(PATH,sep=",",header=True)
    print("--- Finished Stock: ", company)

print("-- End Data Download ")
