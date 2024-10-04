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


PRELABELLING_DIR_PATH = os.path.dirname(os.path.dirname(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.dirname(PRELABELLING_DIR_PATH))

from database.scripts.connection import Connection
from database.scripts.requestdata import RequestData
from aws.scripts.s3datahive import S3DataHive

from prelabelling.models.torchvision.datasets.resnetdataset import ResnetDataset
from prelabelling.samplers.sampler import Sampler
from main import CONNECTION

class BaseSampler(Sampler):
    """
    A higher level sampler that inherits from the Sampler class.
    """

    def __init__(self, 
                 batch_name: str,
                 task_name: str,
                 min_contributions: int = 1,
                 conn: mysql.connector.MySQLConnection = CONNECTION,
                 ) -> None:
        """
        This base sampler handles all the pipelines step without using any data selection 
        strategy, but keeping every datapoint that is already labellized.

        Args
        ----
        - batch_name: the name of the batch to sample data from
        - task_name: the name of the task to sample labels from. The batch must support this task, otherwise
        the sampling won't be executed
        - conn: a MySQLConnection object, that links the classes to the database
        """    
        super().__init__(conn=conn)

        self.conn = conn
        self.batch_name = batch_name
        self.task_name = task_name
        self.min_contributions = min_contributions

        self.request = RequestData(conn=self.conn)
        self.s3client = S3DataHive(conn=self.conn)

        self._init_sampler()


    def apply_strategy(self, df: pd.DataFrame = None):
        """
        The strategy to apply.
        The BaseSampler applies a base strategy, that keeps as many training examples as possible.

        Args
        ----
        - df: the df to read datapoints from

        Returns
        -------
        - datapoint_list: the list of datapoints paths
        - labels_list: the list of labels paths (as jsons)
        - df: the parsed df after applying the strategy
        """

        if df:
            self.df = df
        
        print("BaseSampler >> Applying strategy") # Base strategy does nothing

        self.df.to_csv(os.path.join(os.path.dirname(PRELABELLING_DIR_PATH), "temp", "df_datapoints.csv"), index=False)

        return self.datapoints_from_labels(df=self.df)


    def _init_sampler(self):
        """
        Runs the parent sampler. The strategy will be then applied on this first parsed batch
        """
        self.labels_from_batch(batch_name=self.batch_name, label_task=self.task_name)
        self.df = self.parse_batch_labels(min_contributions=self.min_contributions)

        return self.df