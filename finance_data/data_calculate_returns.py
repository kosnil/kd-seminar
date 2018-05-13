import pandas as pd
import numpy as np

print("-- Start Return Calculation ")

STOCKS = ["BMW.DE","BAS.DE",'AAPL','AIR.DE','TSLA',"BAYN.DE",'TSS',"ALV.DE","VIV",'005930.KS']

for id in STOCKS:
    print("--- Calculate Returns for ", id)
    PATH = "data/"+id+".csv"
    df = pd.read_csv(PATH,sep=",",index_col=0)
    df[id] = np.log(df[id]) - np.log(df[id].shift(1))
    df = df.dropna()
    df.to_csv(PATH, sep=",", header=True)

print("-- End Return Calculation ")
