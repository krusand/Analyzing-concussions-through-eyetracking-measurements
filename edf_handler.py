import os
import glob
import shutil
from tqdm import tqdm


def move_files():
    root_path = "/mnt/c/Users/idwe/Documents/Data/"
    eyetracking_convert_path = root_path + "EyeTrackingData_convert_edf/"
    eyetracking_path = root_path + "EyeTrackingData/"

    eyetracking_folder_files = os.listdir(eyetracking_path)

    for folder in eyetracking_folder_files:
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


def are_all_edf_are_converted():
    root_path = "/mnt/c/Users/idwe/Documents/Data/"
    eyetracking_path = root_path + 'EyetrackingData_convert_edf/'
    eyetracking_files = os.listdir(eyetracking_path)
    print(eyetracking_files)


def main():
    are_all_edf_are_converted()