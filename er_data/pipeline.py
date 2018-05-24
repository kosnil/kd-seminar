import os

os.chdir("er_data")
print("Directory: " , os.getcwd())

if(os.path.exists("data")==False):
    os.mkdir("data")
    print("- Start Download Script")
    os.system("python features.py")
    print("- End Download Script")
else:
    print("- Database already existing")


os.chdir("..")