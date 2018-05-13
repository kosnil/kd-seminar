import os


if(os.path.exists("data")==False):
    print("- Start Download Script")
    os.system("python data_download.py")
    print("- End Download Script")
else:
    print("- Database already existing")

print("- Update Database if necessary")
os.system("python data_update.py")
print("- Update done")