import os
import glob
import shutil
from tqdm import tqdm

root_path = "/mnt/c/Users/idwe/Documents/Data/"
eyetracking_convert_path = root_path + "EyeTrackingData_convert_edf/"
eyetracking_path = root_path + "EyeTrackingData/"

root_folder_files = os.listdir(eyetracking_path)
print(root_folder_files)

for folder in root_folder_files:
    if ".DS_Store" in folder:
            continue
    folder_path = eyetracking_path + folder
    sub_folders = os.listdir(folder_path)
    for sub_folder in tqdm(sub_folders):
        if ".DS_Store" in sub_folder:
            continue
        sub_folder_path = folder_path + "/" + sub_folder
        files = os.listdir(sub_folder_path)
        for file in files:
            if ".DS_Store" in file:
                continue
            file_path = sub_folder_path + "/" + file
            if ".EDF" in file and not file.startswith("."):
                shutil.copyfile(file_path, eyetracking_convert_path + file)
