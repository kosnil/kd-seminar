import os

if(os.path.exists("data")==False):
    os.mkdir("data")
    print("- Start Download Script")
    os.system("python finance_data_download.py")
    print("- End Download Script")
else:
    print("- Database already existing")


print("- Start Data Aggregation Script")
os.system("python finance_data_aggregation.py")
print("- End Data Aggregation Script")

