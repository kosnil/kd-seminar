import requests
import pandas as pd
import os

print("-- Start Data Download ")
API_KEY = "5Z6D01U9OW2CZ202"
STOCKS = ["BMW.DE","BAS.DE",'AAPL','AIR.DE','TSLA',"BAYN.DE",'TSS',"ALV.DE","VIV",'005930.KS']
DIRECTORY_PATH = "data"

if(os.path.exists(DIRECTORY_PATH)==False):
    os.mkdir("data")
else:
    print("Directory already exists!")

for id in STOCKS:
    params = {'function': 'TIME_SERIES_DAILY', 'symbol': id,'outputsize':'full','datatype':'json','apikey':API_KEY}
    raw_data = requests.get("https://www.alphavantage.co/query",params=params)
    print("--- Get Data from ", raw_data.url)

    # Subset JSON for Time-Series column
    # and extract cloesing prices
    print("--- Parse JSON to Dataframe ")
    start = raw_data.text.find("Time Series (Daily)")+len("Time Series (Daily) :")
    end = len(raw_data.text)-1
    df = pd.read_json(raw_data.text[start:end],orient='index')
    df['Timestamp'] = df.index
    df = df.reset_index()
    df = df[['Timestamp','4. close']]
    df.columns = ['Timestamp',id]

    # Save to CSV
    PATH = "data/"+ id + ".csv"
    print("--- Save Dataframe to csv ", PATH )
    df.to_csv(PATH,sep=",",header=True)
    print("--- Finished Stock: ", id)

print("-- End Data Download ")
