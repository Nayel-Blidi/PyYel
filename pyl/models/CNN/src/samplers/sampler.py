import os
import sys
import mysql
import mysql.connector
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import cv2
import json
import abc
from abc import abstractmethod, ABC

PRELABELLING_DIR_PATH = os.path.dirname(os.path.dirname(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.dirname(PRELABELLING_DIR_PATH))

from database.scripts.connection import Connection
from database.scripts.requestdata import RequestData
from aws.scripts.s3datahive import S3DataHive

from prelabelling.models.torchvision.datasets.resnetdataset import ResnetDataset
from main import CONNECTION

class Sampler(ABC):
    """
    The sampler handles to data gathering request made to the database. It follows the strategies
    specified during the Active Learning loop.
    """

    def __init__(self, conn: mysql.connector.MySQLConnection = CONNECTION) -> None:
        """
        This base sampler handles all the pipelines step without using any data selection 
        strategy, but keeping every datapoint that is already labellized.

        Args
        ----
        - conn: a SQL (sqlite3) connection object, that links the classes to the database
        """    

        self.conn = conn
        self.request = RequestData(conn=self.conn)
        self.s3client = S3DataHive(conn=self.conn)

    @abstractmethod
    def _init_sampler(self):
        pass

    @abstractmethod
    def apply_strategy(self):
        pass
    

    def labels_from_batch(self, batch_name: str, label_task: str):
        """
        Retreives the labels relevant for a task from the batch.
        """

        # Checks if the batch name exists
        if self.request.select_where(SELECT="*", FROM="batches", WHERE="batch_name", VALUES=(batch_name,)) == []:
            raise ValueError(f"Sampler >> Batch '{batch_name}' does not exist")
        
        # Checks if the labelling task exists
        with open(os.path.join(os.path.dirname(PRELABELLING_DIR_PATH), "aws", "json", "generator.json"), 'r') as file:
            supported_tasks = list(json.load(file).keys()) 
        if label_task not in supported_tasks:
            raise ValueError("Sampler >> Task {label_task} is not supported")
        self.label_task = label_task

        # Check if the batch is compatible with this labelling task
        self.s3client.download_file(key=f"DataHive/Batches/{batch_name}/metadata.json", path=os.path.join(self.s3client.temp_folder_path, "metadata.json"))
        with open(os.path.join(self.s3client.temp_folder_path, f"metadata.json"), 'r') as file:
            batch_json = dict(json.load(file))
        self._check_batch_task_compatibility(batch_dict=batch_json, task_name=label_task)
        # Retreives the batch options from its metadata
        self.options = batch_json["data"][label_task]["options"]

        # All the existing labels are processed first, so thus datapoints will be retreived from the existing and labellized labels
        all_labels = [label[0] for label in self.request.select_where(SELECT="label_path", FROM="batches_labels", WHERE="batch_name", VALUES=(batch_name,))]
        # Selects the labels files of the right type
        self.labels_keys = []
        for label in all_labels:
            label_path = self.request.select_where(SELECT="label_path", FROM="labels", WHERE="(label_path, label_task)", VALUES=(label, label_task))
            if label_path:
                self.labels_keys.append(label_path[0][0])

        return self.labels_keys, self.options


    def parse_batch_labels(self, 
                           min_contributions: int = 1,
                           options: str = None, 
                           label_task: str = None,
                           labels_keys: list[str] = None):
        """
        Parses a downloaded batch to select the labellized datapoints among the batch. This is done
        by parsing every label files, and keeping only those with labels (classes) matching the given options

        Args
        ----
        - min_contributions: the minimum of times a datapoint must have been labellized with said class to be selected
        - options: the list of classes that must appear on the datapoint to be used as a training example
        - label_task: the type of task
        - labels_keys: the list of keys to download from the cloud
        """
        # To overwrite the label selections
        if labels_keys:
            self.labels_keys = labels_keys
        if options:
            self.options = options
        if label_task:
            self.label_task = label_task
        
        # All the batch labels are dowloaded 
        print("Sampler >> Downloading all the label files")
        self.s3client._empty_folder()
        self.s3client.download_files(keys=self.labels_keys)

        # Only labellized datapoints (i.e. with a label fitting the options) will be kept
        self.df = pd.DataFrame([["", "", "", ""]], columns=["datapoint_key", "datapoint_path", "label_key", "label_path"])
        for idx, label_path in enumerate(self.labels_keys):
            with open(os.path.join(self.s3client.temp_folder_path, f"{idx}.json"), 'r') as file:
                label_json = dict(json.load(file))
            
            # Reading all the labels
            labels = {}
            for label_idx, label_dict in enumerate(label_json["data"].values()):
                # If the datapoint was labellized with a matching option
                if any([cls == label_dict["label"]["cls"] for cls in self.options]) and label_dict["contributions"] >= min_contributions:
                    labels[label_idx] = label_dict["label"]

            if labels:
                # Keeping the matching classes (cls)
                with open(os.path.join(self.s3client.temp_folder_path, f"{idx}.json"), 'w') as file:
                    json.dump(labels, file, indent=4)
                datapoint_key:str = label_json["metadata"]["datapoint_path"]
                datapoint_path = os.path.join(self.s3client.temp_folder_path, f"{idx}{os.path.splitext(os.path.basename(datapoint_key))[1]}")
                label_key = f"{datapoint_key.split('datapoint', maxsplit=1)[0]}{self.label_task}.json"
                label_path = os.path.join(self.s3client.temp_folder_path, f"{idx}.json")
                self.df = pd.concat([self.df, pd.DataFrame([[datapoint_key, datapoint_path, label_key, label_path]], columns=["datapoint_key", "datapoint_path", "label_key", "label_path"])], axis=0)
            else:
                # Else the datapoint is not kept, so the label file is deleted
                self._delete_file(file_path=os.path.join(self.s3client.temp_folder_path, f"{idx}.json"))

        self.df = self.df.iloc[1:, :] # Removing the first init empty row
        return self.df


    def datapoints_from_labels(self, df: pd.DataFrame = None):
        """
        Downloads the datapoints from a pandas dataframe of columns [datapoint_path, datapoint_path, label_path, label_key]. 
        This dataframe links a local datapoint (resp. label) to its remote cloud key.
        """

        if df is not None:
            self.df = df

        # The datapoints are finally downloaded from the cloud
        datapoints_keys = self.df["datapoint_key"].tolist()
        datapoints_paths = self.df["datapoint_path"].tolist()
        print("Sampler >> Downloading the selected datapoints")
        self.s3client.download_files(keys=datapoints_keys, paths=datapoints_paths)

        # The files lists are inferred from the natural order
        # self.datapoints = [file for file in os.listdir(self.s3client.temp_folder_path) if not file.endswith(".json")]
        # self.labels = [file for file in os.listdir(self.s3client.temp_folder_path) if file.endswith(".json")]
        self.datapoints_list = self.df["datapoint_path"].tolist()
        self.labels_list = self.df["label_path"].tolist()
        
        return self.datapoints_list, self.labels_list, self.df


    def split_in_two(self, test_size: float = 0.25, datapoints_list: list[str] = None, labels_list: str = None):
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

        if test_size < 1 and test_size > 0:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.datapoints_list, self.labels_list, test_size=test_size)
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test = [], self.datapoints_list, [], self.labels_list
        return self.X_train, self.X_test, self.Y_train, self.Y_test


    def send_to_dataloader(self, 
                           dataset:Dataset, 
                           data_transform=None, 
                           target_transform=None,
                           chunks: int = 1, 
                           batch_size: int = None, 
                           num_workers: int = 0, 
                           drop_last: bool = True,
                           shuffle: bool = True):
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
                                     data_transform=data_transform, target_transform=target_transform)
        self.test_dataset = dataset(datapoints_list=self.X_test, labels_list=self.Y_test,
                                     data_transform=data_transform, target_transform=target_transform)
        
        # The batch_size parameter has priority over the number of chunks 
        if chunks and not batch_size:
            train_batch_size = self.train_dataset.__len__()//chunks
            test_batch_size = self.test_dataset.__len__()//chunks
        else:
            train_batch_size = batch_size
            test_batch_size = batch_size

        # Dataloader required for the training loop
        if self.train_dataset: 
            self.train_dataloader = DataLoader(self.train_dataset, 
                                            batch_size=train_batch_size, 
                                            shuffle=shuffle, drop_last=drop_last, 
                                            collate_fn=self.train_dataset._collate_fn,
                                            num_workers=num_workers)
        else: # Training is empty to avoid errors
            self.train_dataloader = []
        # Dataloader required for the testing loop
        self.test_dataloader = DataLoader(self.test_dataset, 
                                          batch_size=test_batch_size,
                                          shuffle=shuffle, drop_last=False, 
                                          collate_fn=self.train_dataset._collate_fn,
                                          num_workers=num_workers)

        return self.train_dataloader, self.test_dataloader


    def _get_classes(self, batch_dict: dict, task_name: str):
        """
        Returns the list of unique classes, i.e. the labelling options
        """

        if task_name not in batch_dict["data"].keys():
            print(f"Sampler >> Batch '{batch_dict['metadata']['batch_name']}' is not created to handle {task_name} task")
            return False

    def _delete_file(self, file_path: str):
        """
        Deletes a local file
        """
        try:
            os.remove(file_path)
        except:
            None
        return None

    def _check_batch_task_compatibility(self, batch_dict: dict, task_name: str):
        """
        Checks if a batch is compatible with the chosen labelling task
        """

        if task_name not in batch_dict["data"].keys():
            print(f"Sampler >> Batch '{batch_dict['metadata']['batch_name']}' is not created to handle {task_name} task")
            return False

        return True
