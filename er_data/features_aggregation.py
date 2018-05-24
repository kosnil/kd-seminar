import os
import pandas as pd

actual_path = os.getcwd()
data_path = actual_path + "/data"

available_datasets = os.listdir(data_path)
print("Datasets " ,available_datasets )
i = 0
for dataset in available_datasets:
    print("Data", dataset)
    PATH = "data/" + dataset
    df = pd.read_csv(PATH, sep=",", header=0, index_col=0)

    if i == 0:
        complete_df = df
        i = 1
    else:
        print("append")
        complete_df = complete_df.append(df)

print(complete_df)
complete_df = complete_df.drop_duplicates()
complete_df = complete_df.sort('Timestamp')
print(complete_df)

complete_df.to_csv("data/sentiment_features_complete.csv",sep=",",header=True)
#df = pd.read_csv('er_data/data/sentiment_features_2018-05-01_2018_05_20.csv',header=0,index_col=0)
#df = df.drop('ibmSentiment',axis=1)
#df.to_csv('er_data/data/sentiment_features_2018-05-01_2018_05_20.csv',header=True)
