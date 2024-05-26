import os
import sys

import numpy as np
import pandas as pd


PRELABELLING_DIR_PATH = os.path.dirname(__file__)
DATABASE_DIR_PATH = os.path.join(os.path.dirname(PRELABELLING_DIR_PATH), "database")

if __name__ == "__main__":
    sys.path.append(os.path.dirname(PRELABELLING_DIR_PATH))

from database.scripts.connection import ConnectionSQLite
from database.scripts.requestdata import RequestData
from database.scripts.createtables import CreateTables
from database.scripts.insertdata import InsertData
from database.scripts.removedata import RemoveData

from gui.AppGUILauncher import AppGUILauncher

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
        



if __name__ == "__main__" and "db" in sys.argv:
    # Starts the DB manager GUI
    app = AppGUILauncher()
    app.start_LabellingToolDBManager()


if __name__ == "__main__" and "africa" in sys.argv:
    # Parses and uploads the AfricaWildlife dataset into the DB 
    folders_list = [os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "buffalo"),
                    os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "rhino"),
                    os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "elephant"),
                    os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "zebra")]
    output_folder = os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "merged")
    merge_into_one_folder(folders_list, output_folder, extensions=["jpg", "txt"], condition="")

    coo = ConnectionSQLite()
    conn = coo.connect_database()
    ins = InsertData(conn=conn)
    rqs = RequestData(conn=conn)
    crt = CreateTables(conn=conn)
    crt.setup_database()

    test_path = os.path.join(os.path.dirname(os.path.dirname(DATABASE_DIR_PATH)), "Datasets", "AfricaWildlife")

    for animal in ["buffalo", "elephant", "rhino", "zebra"]:
        ins.upload_dataset(dataset_type="Image_datapoints",
                           extensions=["png", "jpg", "JPG", "PNG"],
                           dataset_name=f"AfricaWildlife_{animal}",
                           dataset_description=f"The {animal} of the dataset and their labelled positions.",
                           path=os.path.join(test_path, animal))
        
        ins._image_txt_labels(folder_path=f"{test_path}/{animal}",
                              source_dataset=f"AfricaWildlife_{animal}",
                              class_txt={0:"buffalo", 1:"elephant", 2:"rhino", 3:"zebra"})

        ins.upload_labelset(table_name="Image_classification",
                            dataset_name=f"AfricaWildlife_{animal}",
                            columns=["datapoint_key", "confidence", "contributions", "class", "class_txt"],
                            labelset_path=os.path.join(test_path, animal, "labels.csv"))

        ins.upload_labelset(table_name="Image_detection",
                            dataset_name=f"AfricaWildlife_{animal}",
                            columns=["datapoint_key", "confidence", "contributions", "class", "class_txt", "x_min", "x_max", "y_min", "y_max"],
                            labelset_path=os.path.join(test_path, animal, "labels.csv"))

    coo.close_database()

if __name__ == "__main__" and "leaf" in sys.argv:
    # Parses and uploads the LeafDisease dataset into the DB 
    folders_list = [os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "LeafDisease", "images"),
                    os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "LeafDisease", "masks")]
    output_folder = os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "LeafDisease", "merged")
    join_into_one_folder(folders_list, output_folder, extensions=["jpg", "png", "JPG", "PNG"])

    coo = ConnectionSQLite()
    conn = coo.connect_database()
    ins = InsertData(conn=conn)
    rqs = RequestData(conn=conn)
    crt = CreateTables(conn=conn)
    crt.setup_database()

    test_path = os.path.join(os.path.dirname(os.path.dirname(DATABASE_DIR_PATH)), "Datasets", "LeafDisease")

    ins.upload_dataset(dataset_type="Image_datapoints",
                       extensions=["png", "jpg", "JPG", "PNG"],
                       dataset_name=f"LeafDisease",
                       dataset_description=f"A sick leaf with its infected areas highlighted",
                       path=os.path.join(test_path, "merged"))
    
    ins._image_mask_labels(folder_path=f"{test_path}/merged",
                          source_dataset=f"LeafDisease",
                          class_txt="sick leaf")

    ins.upload_labelset(table_name="Image_segmentation",
                        dataset_name=f"LeafDisease",
                        columns=["datapoint_key", "confidence", "contributions", "label_path", "class_txt"],
                        labelset_path=os.path.join(test_path, "merged", "labels.csv"))

    coo.close_database()

if __name__ == "__main__" and "fish" in sys.argv:
    # Parses and uploads the FishSegmentation dataset into the DB 

    coo = ConnectionSQLite()
    conn = coo.connect_database()
    ins = InsertData(conn=conn)
    rqs = RequestData(conn=conn)
    crt = CreateTables(conn=conn)
    crt.setup_database()

    fish_names_list = ["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", 
                       "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet", "Trout"]
    for fish_name in fish_names_list:
        folders_list = [os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "FishSegmentation", fish_name, f"{fish_name}"),
                        os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "FishSegmentation", fish_name, f"{fish_name} GT")]
        output_folder = os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "FishSegmentation", fish_name, "merged")
        join_into_one_folder(folders_list, output_folder, extensions=["jpg", "png", "JPG", "PNG"])

        test_path = os.path.join(os.path.dirname(os.path.dirname(DATABASE_DIR_PATH)), "Datasets", "FishSegmentation")

        ins.upload_dataset(dataset_type="Image_datapoints",
                           extensions=["png", "jpg", "JPG", "PNG"],
                           dataset_name=f"FishSegmentation_{fish_name.replace(' ', '_')}",
                           dataset_description=f"The {fish_name} of the dataset and their labelled positions.",
                           path=os.path.join(test_path, fish_name, "merged"))
        
        ins._image_mask_labels(folder_path=os.path.join(test_path, fish_name, "merged"),
                              source_dataset=f"FishSegmentation_{fish_name.replace(' ', '_')}",
                              class_txt=fish_name,
                              extensions=["png", "jpg", "JPG", "PNG"])

        ins.upload_labelset(table_name="Image_segmentation",
                            dataset_name=f"FishSegmentation_{fish_name.replace(' ', '_')}",
                            columns=["datapoint_key", "confidence", "contributions", "label_path", "class_txt"],
                            labelset_path=os.path.join(test_path, fish_name, "merged", "labels.csv"))

    coo.close_database()

if __name__ == "__main__" and "breast" in sys.argv:
    # Parses and uploads the LeafDisease dataset into the DB 
    folders_list = [os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "BreastCancer", "benign"),
                    os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "BreastCancer", "malignant"),
                    os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "BreastCancer", "normal")]
    rename_folder_content([folders_list[-1]], add="GT_", remove=5, extensions=["jpg", "png", "JPG", "PNG"], condition="mask")

    coo = ConnectionSQLite()
    conn = coo.connect_database()
    ins = InsertData(conn=conn)
    rqs = RequestData(conn=conn)
    crt = CreateTables(conn=conn)
    crt.setup_database()

    rmv = RemoveData(conn=conn)
    rmv._erase_labels(labels_table=["Image_classification", "Image_segmentation"], datapoints_keys=np.arange(13445, 14224))

    test_path = os.path.join(os.path.dirname(os.path.dirname(DATABASE_DIR_PATH)), "Datasets", "BreastCancer")

    for cell in ["benign", "malignant", "normal"]:
        ins.upload_dataset(dataset_type="Image_datapoints",
                           extensions=["png", "jpg", "JPG", "PNG"],
                           dataset_name=f"BreastCancer_{cell}",
                           dataset_description=f"A collection of female breast {cell} cells",
                           path=os.path.join(test_path, cell))
        
        ins._image_mask_labels(folder_path=os.path.join(test_path, cell),
                              source_dataset=f"BreastCancer_{cell}",
                              class_txt=cell,
                              extensions=["png", "PNG"])

        ins.upload_labelset(table_name="Image_classification",
                            dataset_name=f"BreastCancer_{cell}",
                            columns=["datapoint_key", "confidence", "contributions", "class_txt"],
                            labelset_path=os.path.join(test_path, cell, "labels.csv"))

        ins.upload_labelset(table_name="Image_segmentation",
                            dataset_name=f"BreastCancer_{cell}",
                            columns=["datapoint_key", "confidence", "contributions", "label_path", "class_txt"],
                            labelset_path=os.path.join(test_path, cell, "labels.csv"))

    coo.close_database()


if __name__ == "__main__" and "human" in sys.argv:
    # Parses and uploads the LeafDisease dataset into the DB 
    folders_list = [os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "LeafDisease", "images"),
                    os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "LeafDisease", "masks")]
    output_folder = os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "LeafDisease", "merged")
    join_into_one_folder(folders_list, output_folder, extensions=["jpg", "png", "JPG", "PNG"])

    coo = ConnectionSQLite()
    conn = coo.connect_database()
    ins = InsertData(conn=conn)
    rqs = RequestData(conn=conn)
    crt = CreateTables(conn=conn)
    crt.setup_database()

    test_path = os.path.join(os.path.dirname(os.path.dirname(DATABASE_DIR_PATH)), "Datasets", "LeafDisease")

    ins.upload_dataset(dataset_type="Image_datapoints",
                       extensions=["png", "jpg", "JPG", "PNG"],
                       dataset_name=f"LeafDisease",
                       dataset_description=f"A sick leaf with its infected areas highlighted",
                       path=f"{test_path}/merged")
    
    ins._image_mask_labels(folder_path=f"{test_path}/merged",
                          source_dataset=f"LeafDisease",
                          class_txt="N/A")

    ins.upload_labelset(table_name="Image_segmentation",
                        columns=["datapoint_key", "confidence", "contributions", "label_path", "class_txt"],
                        labelset_path=f"{test_path}/merged/labels.csv")

    coo.close_database()