import requests
import pandas as pd
import numpy as np

print("-- Start Data Update ")


API_KEY = "5Z6D01U9OW2CZ202"
STOCKS = ["BMW.DE","BAS.DE",'AAPL','AIR.DE','TSLA',"BAYN.DE",'TSS',"ALV.DE","VIV",'005930.KS']

for id in STOCKS:
    # Get current content of database
    PATH = "data/"+ id + ".csv"
    database = pd.read_csv(PATH,sep=",",index_col=0,parse_dates=["Timestamp"])
    max_date = database['Timestamp'].max()

    # Get new data from API
    params = {'function': 'TIME_SERIES_DAILY', 'symbol': id,'outputsize':'compact','datatype':'json','apikey':API_KEY}
    raw_data = requests.get("https://www.alphavantage.co/query",params=params)
    print("--- Get Data from ", raw_data.url)

    # Subset JSON for Time-Series column
    # and extract cloesing prices
    print("--- Parse JSON to Dataframe ")
    start = raw_data.text.find("Time Series (Daily)")+len("Time Series (Daily) :")
    end = len(raw_data.text)-1
    df = pd.read_json(raw_data.text[start:end],orient='index')
    df['Timestamp'] = df.index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d')
    df = df.reset_index()
    df = df[['Timestamp','4. close']]
    df.columns = ['Timestamp',id]
    df[id] = np.log(df[id]) - np.log(df[id].shift(1))
    df = df.dropna()
    df = df.reset_index(drop=True)

    new_max_date = df['Timestamp'].max()

    if(new_max_date>max_date):
        print("--- Append additional rows to Database")
        database = database.append(df.iloc[np.where(df['Timestamp']>max_date)])
        database = database.reset_index(drop=True)

        # Save to CSV
        print("--- Save Database to csv ", PATH)
        database.to_csv(PATH, sep=",", header=True)
    else:
        print("--- Database for ", id, " already up to date! ")

    print("--- Finished Stock: ", id)


print("-- End Data Update ")
