import os

os.chdir("finance_data")
print("Directory: " , os.getcwd())

if(os.path.exists("data")==False):
    os.mkdir("data")
    print("- Start Download Script")
    os.system("python data_download.py")
    print("- End Download Script")
else:
    print("- Database already existing")


print("- Start Database Update Script")
os.system("python data_update.py")
print("- End Database Update Script")

print("- Start Data Aggregation Script")
os.system("python data_aggregation.py")
print("- End Data Aggregation Script")

os.chdir("..")