import os
import glob
import shutil
from tqdm import tqdm


def copy_files():
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


def move_files_to_be_re_encoded():
    files = get_files_to_be_re_encoded()
    for file in files:
        shutil.move(src="/mnt/c/Users/idwe/Documents/Data/EyetrackingData_convert_edf/" + file,
                    dst="/mnt/c/Users/idwe/Documents/Data/EyetrackingData_convert_edf/missing/" + file)

def get_metadata_file_paths(root_path):
    metadata_files = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith(".toml") and not file.startswith("._"):
                metadata_files.append([root + "/", file])
    return metadata_files

def copy_metadata_files(file_paths):
    for root, file in file_paths:
        print(f"Copying file: {file}")
        shutil.copyfile(src=root + file, 
                        dst="/mnt/c/Users/idwe/Documents/Github/Analyzing-concussions-through-eyetracking-measurements/data/metadata/raw/" + file)

def main():
    #move_files_to_be_re_encoded()
    metadata_files = get_metadata_file_paths(root_path="/mnt/c/Users/idwe/Documents/Data/EyetrackingData/")
    copy_metadata_files(metadata_files)

if __name__ == "__main__":
    main()