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


def get_files_to_be_re_encoded():
    root_path = "/mnt/c/Users/idwe/Documents/Data/"
    eyetracking_path = root_path + 'EyetrackingData_convert_edf/'
    eyetracking_files = os.listdir(eyetracking_path)
    edf_files = {file[:-4] for file in eyetracking_files if file.endswith(".EDF")}
    asc_files = {file[:-4] for file in eyetracking_files if file.endswith(".asc")}
    return [file + ".EDF" for file in (edf_files - asc_files)]

def move_file(file_name):
    root_path = "/mnt/c/Users/idwe/Documents/Data/"
    eyetracking_path = root_path + 'EyetrackingData_convert_edf/'
    eyetracking_missing_path = eyetracking_path + "missing/"
    shutil.move(src=eyetracking_path + file_name,
                dst=eyetracking_missing_path + file_name)

def move_files_to_be_re_encoded():
    files = get_files_to_be_re_encoded()
    for file in files:
        move_file(file)

def metadata():
    


def main():
    #move_files_to_be_re_encoded()
    get_files_to_be_re_encoded()

if __name__ == "__main__":
    main()