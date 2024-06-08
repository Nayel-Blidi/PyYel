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


class Sampler():
    """
    The sampler handles to data gathering request made to the database. It follows the strategies
    specified during the Active Learning loop.
    """

    def __init__(self, df:pd.DataFrame, device:str="cpu") -> None:
        """        
        Args
        ----
        - df: the pandas dataframe (csv file) featuring the datapoints paths and its associated labels
        - device: the device to send the data to (default is ``"cpu"``)
        """    
        self.df = df
        self.device = device


    def load_from_df(self, labels_type:str):
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
            labels = "class_txt"
            shape = (len(self.df), 1)
        elif labels_type == "Image_detection":
            labels = "x_min, y_min, x_max, y_max, class_txt"
            shape = (len(self.df), 5)
        elif labels_type == "Image_segmentation":
            labels = "label_path, class_txt"
            shape = (len(self.df), 2)
        else:
            raise ValueError(f"Task {labels_type} is not supported")

        self.datapoints_list = self.df["datapoint"].to_numpy()
        self.labels_list = self.df[labels].to_numpy()
        self.labels_list = self.labels_list.reshape(shape)

        # The classes as text are required to edit a label_encoder if required by the model
        unique_txt_classes = list(set(self.df["class_txt"].to_list()))

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
        if datapoints_list is not None:
            self.datapoints_list = datapoints_list
        if labels_list is not None:
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

