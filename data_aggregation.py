import os
import pandas as pd
import numpy as np

'''
Aggregate Results to final df
'''
if(os.path.exists("final_data")==False):
    os.mkdir("final_data")


print("- Start Aggregate! ")
print("-- Import Datasets")
er_data = pd.read_csv("er_data/data/sentiment_features_complete.csv",index_col=0,header=0,parse_dates=["Timestamp"])
stock_data = pd.read_csv('finance_data/data/aggregated_returns.csv',index_col=0,header=0,parse_dates=['Timestamp'])

print("-- Merge Datasets")

df = er_data.merge(stock_data,on='Timestamp')
df['Previous_Day_Return'] = np.zeros(df.shape[0])
df['Next_Day_Return'] = np.zeros(df.shape[0])


print("-- Adjust Returns")
companies = ['Samsung','BASF','Apple','Tesla','Airbus','Bayer','BMW','Telefonica','Google','Allianz','Total']
for company in companies:
    df.loc[df.ID==company,'Previous_Day_Return'] = df.loc[df.ID==company,company]
    df.loc[df.ID == company, 'Next_Day_Return'] = df.loc[df.ID == company, company].shift(-1)

# For Checking purposes
#df.loc[df.ID=='Apple',['Timestamp','Previous_Day_Return','Apple','Next_Day_Return']]

df = df.drop(companies,axis=1)
df = df.dropna()

print("-- Export Datasets")
df.to_csv("final_data/complete_data.csv", sep=",", header=True)

print("- Finish Aggregation! ")
