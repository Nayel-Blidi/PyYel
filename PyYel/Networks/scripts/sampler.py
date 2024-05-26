import os
import sys
import sqlite3
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import cv2

PRELABELLING_DIR_PATH = os.path.dirname(os.path.dirname(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.dirname(PRELABELLING_DIR_PATH))

from database.scripts.connection import ConnectionSQLite
from database.scripts.requestdata import RequestData

from prelabelling.models.torchvision.datasets.resnetdataset import ResnetDataset

class Sampler():
    """
    The sampler handles to data gathering request made to the database. It follows the strategies
    specified during the Active Learning loop.
    """

    def __init__(self, conn:sqlite3.Connection, device:str="cpu") -> None:
        """        
        Args
        ----
        - conn: a SQL (sqlite3) connection object, that links the classes to the database
        - device: the device to send the data to (default is ``<cpu>``)
        """    
        self.conn = conn
        self.device = device
        self.request = RequestData(conn=self.conn)


    def load_from_db(self, datapoints_type:str, subdataset_name:str, labels_type:str):
        """
        Loads the datapoints paths and labels related to a subdatset saved on the database.

        Args
        ----
        - datapoints_type: the name of the table to pick the data from (eg: Image_datapoints)
        - subdataset_name: the name of the SubDataset table that references the datapoints_keys of said SubDataset
        - labels_type: The task (labels table) to select the labels from, i.e. the task to execute (relevant
        columns are automatically infered from the database architecture)

        Returns
        -------
        - datapoints_list: the list of paths to the datapoints saved on the server. These paths will be used by the
        dataloader when applying the ``__getitem__`` method to send the batch data into the computing ``device``

        >>> datapoints_list
        >>> ["C:/.../image0.png", "C:/.../image1.png", ...]

        - labels_list: the list of tuples where each item is a row from the <labels_type> table. 
        ``datapoint_key`` will always be the first element of each item.

        >>> labels_list
        >>> [(datapoint_key1:int, class_int1:int, "class_txt1"), (datapoint_key1:int, class_int2:int, "class_txt2"), (datapoint_key2:int, class_int1:int, "class_txt1"),...]
        >>> labels_list
        >>> [(datapoint_key1:int, x_min1:float, y_min1:float, x_max1:float, y_max1:float, "class_txt1"), (datapoint_key1:int, x_min2:float, y_min2:float, x_max2:float, y_max2:float, "class_txt2"), ...]
        """

        if labels_type == "Image_classification":
            labels = "class, class_txt"
        elif labels_type == "Image_detection":
            labels = "class, x_min, y_min, x_max, y_max, class_txt"
        elif labels_type == "Image_segmentation":
            labels = "label_path, class_txt"
        else:
            raise ValueError(f"Task {labels_type} not supported")

        datapoints_keys = self.request.select_keys_from_subdataset(subdataset_name=subdataset_name)
        self.datapoints_list = self.request.select_paths_from_keys(datapoints_keys=datapoints_keys, 
                                                                   datapoints_type=datapoints_type)
        self.labels_list = self.request.select_labels_from_keys(datapoints_keys=datapoints_keys, 
                                                                labels_type=labels_type, 
                                                                labels=labels)

        # The classes as text are required to edit a label_encoder if required by the model
        unique_txt_classes = list(set([row[-1] for row in self.labels_list]))

        # The datapoints and labels rows are grouped into a dictionnary by sorted keys, so it ensures the connection between a datapoints and its label(s)
        self.datapoints_list = {id: [tup[1:] for tup in self.datapoints_list if tup[0] == id] for id in set(tup[0] for tup in self.datapoints_list)}
        self.datapoints_list = {k: self.datapoints_list[k] for k in sorted(self.datapoints_list.keys())}

        self.labels_list = {id: [tup[1:] for tup in self.labels_list if tup[0] == id] for id in set(tup[0] for tup in self.labels_list)}
        self.labels_list = {k: self.labels_list[k] for k in sorted(self.labels_list.keys())}
        
        # The dictionnaries are converted into list of str (datapoint_path) and list of arrays (labels)
        self.datapoints_list = [tup[0][0] for tup in self.datapoints_list.values()] # list of singleton to list of elements
        self.labels_list = list({key: np.array(list_of_tuples, dtype=object) for key, list_of_tuples in self.labels_list.items()}.values())

        return self.datapoints_list, self.labels_list, unique_txt_classes


    def split_in_two(self, test_size=0.25, datapoints_list=None, labels_list=None):
        """
        Splits the querried data into a training and testing batch. Can also be used out of the sampling
        pipeline as a util by overwriting the ``<datapoints_list>`` and/or ``<labels_list>`` inputs.

        Args
        ----
        - test_size: the percentage of batch data to allocate to the testing loop. Thus it won't be used during 
        the whole training process.
        - datapoints_list: the list of paths as described in the ``<from_DB>`` method 
        - labels_list: the list of label tuples as described in the ``<from_DB>`` method 
        """
        if datapoints_list:
            self.datapoints_list = datapoints_list
        if labels_list:
            self.labels_list = labels_list

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.datapoints_list, self.labels_list, test_size=test_size)
        return self.X_train, self.X_test, self.Y_train, self.Y_test


    def send_to_dataloader(self, dataset:Dataset, 
                           data_transform=None, target_transform=None,
                           chunks:int=1, batch_size:int=None, drop_last=True,
                           num_workers=0):
        """
        Returns a training and testing dataloaders objects from the sampled ``datapoints_list`` and ``labels_list``.

        Args
        ----
        - dataset: a torch Dataset subclass that is compatible with the performed task (a forciori the loaded model)
        - transform: a short datapoints preprocessing pipeline, that should be model specific 
        (such as resizing an image input, or vectorizing a word...) and data specific (normalizing...)
        - chunks: the number of batch to divide the dataset into
        """

        # Custom datasets
        self.train_dataset = dataset(datapoints_list=self.X_train, labels_list=self.Y_train, 
                                     data_transform=data_transform, target_transform=target_transform,
                                     device=self.device)
        self.test_dataset = dataset(datapoints_list=self.X_test, labels_list=self.Y_test,
                                     data_transform=data_transform, target_transform=target_transform,
                                     device=self.device)

        # The batch_size parameter has priority over the number of chunks 
        if chunks and not batch_size:
            train_batch_size = self.train_dataset.__len__()//chunks
            test_batch_size = self.test_dataset.__len__()//chunks
        else:
            train_batch_size = batch_size
            test_batch_size = batch_size

        # Dataloader required for the training loop
        self.train_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=train_batch_size, 
                                           shuffle=True, drop_last=drop_last, 
                                           collate_fn=self.train_dataset._collate_fn,
                                           num_workers=num_workers)
        # Dataloader required for the testing loop
        self.test_dataloader = DataLoader(self.test_dataset, 
                                          batch_size=test_batch_size,
                                          shuffle=True, drop_last=False, 
                                          collate_fn=self.train_dataset._collate_fn,
                                          num_workers=num_workers)

        return self.train_dataloader, self.test_dataloader

