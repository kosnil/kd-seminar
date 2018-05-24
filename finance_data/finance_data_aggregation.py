import pandas as pd

print("-- Start Data Aggregation ")

STOCKS = ["ABEA.BE","BMW.DE","BAS.DE",'AAPL','AIR.DE','TSLA',"BAYN.DE",'TSS',"ALV.DE","VIV",'005930.KS']
stock_names = {"ABEA.BE":"Google","TSS":"Total","AIR.DE":"Airbus","BMW.DE":"BMW","BAS.DE":"BASF","AAPL":"Apple","TSLA":"Tesla","BAYN.DE":"Bayer","ALV.DE":"Allianz","VIV":"Telefonica","005930.KS":"Samsung"}



i = 0
for id in STOCKS:
    company = stock_names[id]
    PATH = "data/" + company + ".csv"
    df = pd.read_csv(PATH,sep=",",index_col=0)
    print("-- Stock Time Range", company , "From", df['Timestamp'].min(), " to " , df['Timestamp'].max())

    if i==0:
        database = df
        i = 1
    else:
        database = pd.merge(df,database,on="Timestamp")

    print("-- Database Time Range","From", database['Timestamp'].min(), " to " , database['Timestamp'].max())

database.to_csv("data/aggregated_returns.csv", sep=",", header=True)

print("-- End Data Aggregation ")
