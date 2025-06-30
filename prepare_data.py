import os

data_dir = "data"
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f"Files in '{data_dir}':", files)
else:
    print(f"Directory '{data_dir}' does not exist.")
