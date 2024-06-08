import os
import sys

import numpy as np
import pandas as pd
import cv2

DATA_DIR_PATH = os.path.dirname(__file__)
if __name__ == "__main__":
    sys.path.append(os.path.dirname(DATA_DIR_PATH))

import shutil
from tqdm import tqdm



def merge_into_one_folder(input_folders:list[str], output_folder:str, extensions=["jpg", "txt"], condition:str=""):
    """
    Takes a list of folders to merge into one.
    These folders a composed of both datapoints (extension[0]) and labels (extension[1]) files.

    To merge a folder of datapoints (extension[0]) and an other one of labels (extension[1]), refer to
    the join_into_one_folder() function.

    - input_folder: the list of folders path to merge the content together
    - output_folder: the folder path to save the merge to. If non existent, creates it
    - extensions: a list of two extensions, to specify which one count as datapoints and which one count as labels. If the 
    extensions are idenctical, you should use the <condition> to specify the label name condition
    - condition: a file ending condition to discriminate labels from the datapoints

    Eg:
    >>> files_list = [image1.png, image1_mask.png...]
    >>> merge_into_one_folder(input_folders=, output_folder=, extensions=["png", "png"], condition="mask")

    """

    c = 0
    dict = {}
    for folder in input_folders:
        for file in os.listdir(folder):
            if file.endswith(extensions[0]):
                dict[c] = (os.path.join(folder, f"{os.path.basename(file)[:-4]}.{extensions[0]}"), os.path.join(folder, f"{os.path.basename(file)[:-4]}{condition}.{extensions[1]}"))
                c+=1

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Copying:", dict[0])
    for cc in tqdm(range(c)):
        shutil.copy2(dict[cc][0], os.path.join(output_folder, f"{cc}.{extensions[0]}"))
        shutil.copy2(dict[cc][1], os.path.join(output_folder, f"GT_{cc}.{extensions[1]}"))

    return None

def join_into_one_folder(input_folders:list[str], output_folder:str, extensions:tuple=("jpg", "txt")):
    """
    Takes a list of one datapoints (extension[0]) folder and an other one of labels (extension[1]) and merge it together.

    To merge mixed folders into one, refer to the merge_into_one_folder() function.

    - input_folder: the list of the two folders path to merge the content together
    """

    l = []
    for folder in input_folders:
        l.append([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(tuple(extensions))])

    l = list(zip(*l))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Copying:", l[0])
    for idx in tqdm(range(len(l))):
        shutil.copy2(l[idx][0], os.path.join(output_folder, f"{idx}.{extensions[0]}"))
        shutil.copy2(l[idx][1], os.path.join(output_folder, f"GT_{idx}.{extensions[1]}"))

    return None

def rename_folder_content(folder_list:list, add:str="GT_", remove:int=0, extensions:list=["png"], condition:str=""):

    for folder in folder_list:
        for file in os.listdir(folder):
            if file.endswith(tuple([f"{condition}.{extension}" for extension in extensions])):
                os.rename(os.path.join(folder, file), os.path.join(folder, f"{add}{os.path.splitext(file)[0][:-remove]}{os.path.splitext(file)[1]}")) # os join(path, add+file_name[:-remove].file_extension)
    
    return None
        

def empty_folder_content(folder_path, extensions=['.png', '.txt']):
    c=0
    for root, dirs, files in os.walk(folder_path):
        print(f"Deleting files from {folder_path}.")
        for file in tqdm(files):
            if file.endswith(tuple(extensions)):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                c+=1
    print(f"{c} files deleted from {folder_path}.")
    return None

def open_image(image_path):
    return cv2.imread(image_path)

def open_image_folder(folder_path, extensions=[".png", ".jpg"]):
    print(f"Reading files from '{folder_path}' and opening those ending with {extensions}.")
    image_list = [open_image(image_path=os.path.join(folder_path, file)) for file in tqdm(os.listdir(folder_path)) if file.endswith(tuple(extensions))]
    # image_list = [print(os.path.join(folder_path, file)) for file in tqdm(os.listdir(folder_path)) if file.endswith(tuple(extensions))]
    return image_list
