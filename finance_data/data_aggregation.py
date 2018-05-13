import pandas as pd

print("-- Start Data Aggregation ")

STOCKS = ["BAS.DE",'AAPL','AIR.DE','TSLA',"BAYN.DE",'TSS',"ALV.DE","VIV",'005930.KS']

PATH = "data/" + "BMW.DE" + ".csv"
database = pd.read_csv(PATH, sep=",",index_col=0)

for id in STOCKS:
    PATH = "data/" + id + ".csv"
    df = pd.read_csv(PATH,sep=",",index_col=0)
    database = pd.merge(df,database,on="Timestamp")

database.to_csv("data/aggregated_returns.csv", sep=",", header=True)

print("-- End Data Aggregation ")
