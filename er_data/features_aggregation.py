import os
import pandas as pd

actual_path = os.getcwd()
data_path = actual_path + "/data"
RESULT_PATH = "data/sentiment_features_complete.csv"


available_datasets = os.listdir(data_path)
print("- Datasets " ,available_datasets )
i = 0
for dataset in available_datasets:
    print("-- Import Data", dataset)
    PATH = "data/" + dataset
    df = pd.read_csv(PATH, sep=",", header=0, index_col=0)

    if i == 0:
        complete_df = df
        i = 1
    else:
        print("-- Append Data")
        complete_df = complete_df.append(df)

complete_df = complete_df.drop_duplicates(subset=['Timestamp','ID'])
complete_df = complete_df.sort_values('Timestamp')
complete_df.reset_index(drop=True,inplace=True)
print("- Save dataset")
complete_df.to_csv(RESULT_PATH,sep=",",header=True)
#df = pd.read_csv('er_data/data/sentiment_features_2018-05-01_2018_05_20.csv',header=0,index_col=0)
#df = df.drop('ibmSentiment',axis=1)
#df.to_csv('er_data/data/sentiment_features_2018-05-01_2018_05_20.csv',header=True)
