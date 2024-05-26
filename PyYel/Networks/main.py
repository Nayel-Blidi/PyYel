import os
import sys

import numpy as np
import pandas as pd


PRELABELLING_DIR_PATH = os.path.dirname(__file__)

if __name__ == "__main__":
    sys.path.append(os.path.dirname(PRELABELLING_DIR_PATH))

from database.scripts.connection import ConnectionSQLite
from database.scripts.requestdata import RequestData

from prelabelling.scripts.sampler import Sampler
from prelabelling.scripts.compiler import Compiler
from prelabelling.scripts.labeller import labeller

from prelabelling.models.torchvision.classificationRESNET import ClassificationRESNET

import shutil
from tqdm import tqdm

def merge_into_one_folder(input_folders:list[str], output_folder:str):

    c = 0
    dict = {}
    for folder in input_folders:
        for file in os.listdir(folder):
            if file.endswith('.jpg'):
                dict[c] = (os.path.join(folder, f"{os.path.basename(file)[:-4]}.jpg"), os.path.join(folder, f"{os.path.basename(file)[:-4]}.txt"))
                c+=1

    print(dict[0])
    for cc in tqdm(range(c)):
        # print(dict[cc][0])
        # print(os.path.join(output_folder, f"{cc}.jpg"))
        shutil.copy2(dict[cc][0], os.path.join(output_folder, f"{cc}.jpg"))
        shutil.copy2(dict[cc][1], os.path.join(output_folder, f"{cc}.txt"))
        
    return None
    
folders_list = [os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "buffalo"),
                os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "rhino"),
                os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "elephant"),
                os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "zebra")]

merge_into_one_folder(folders_list, os.path.join(os.path.dirname(os.path.dirname(PRELABELLING_DIR_PATH)), "Datasets", "AfricaWildlife", "merged"))

# coo = ConnectionSQLite()
# conn = coo.connect_database()
# request = RequestData(conn=conn)

# sampler = Sampler(conn=conn, request=request)
# sampler.from_folder(folder_path="C:\\Users\\nblidi\\Projets\\Datasets\\AfricaWildlife\\elephant")
# sampler.split()
# train, test = sampler.dataload()

# compiler = Compiler(ClassificationRESNET, version="18")
# compiler.loader()
# compiler.tester()