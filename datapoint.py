
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Datapoint():

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

        self.batch_size =       None
        self.in_channels =      None
        self.height =           None
        self.width =            None
        self.output_size =      None
        self.train_dataloader = None
        self.test_dataloader =  None

        self._datapointShapes()
        self.kwargs =           None

    def getKwargs(self):
        self.kwargs= {
            "batch_size": self.batch_size, 
            "in_channels": self.in_channels, 
            "height": self.height, 
            "width": self.width, 
            "input_size": self.in_channels,
            "output_size": self.output_size,
            # "train_dataloader": self.train_dataloader, 
            # "test_dataloader": self.test_dataloader,
        }
        self.kwargs = {key: value for key, value in self.kwargs.items() if value is not None}
        return self.kwargs
    
    def _datapointShapes(self, display=False) -> tuple:
        if len(self.X.shape) == 1:
            # Data is supposed to be a column of values
            self.batch_size = self.X.shape[0]
            self.in_channels = 1
            self.height = 1
            self.width = 1
        elif len(self.X.shape) == 2:
            # Data is supposed to be a structured frame
            self.batch_size = self.X.shape[0]
            self.in_channels = self.X.shape[1]
            self.height = 1
            self.width = 1
        elif len(self.X.shape) == 3:
            # Data is supposed to be a gray scale image, or layers of structured data like
            self.batch_size = self.X.shape[0]
            self.in_channels = 1
            self.height = self.X.shape[1]
            self.width = self.X.shape[2]
            self.X = np.expand_dims(self.X, axis=1)
        elif len(self.X.shape) == 4:
            # Data is supposed to be rgb images like
            self.batch_size = self.X.shape[0]
            self.in_channels = self.X.shape[1]
            self.height = self.X.shape[2]
            self.width = self.X.shape[3]

        self.output_size = self.y.shape[-1]
        
        if display:
            print("> batch_size, in_channels, height, width, output_size:")
            print("\t", self.batch_size, "\t" , self.in_channels, "\t", self.height, "\t", self.width, "\t", self.output_size)
            
        return self.batch_size, self.in_channels, self.height, self.width, self.output_size

    def runPipeline(self):
        datapoint_pipeline = [
            self.split(),
            self.tensorize(),
            self.normalize(),
            self.dataload(),
        ]
        for step in datapoint_pipeline:
            step
        return None

    def getRawData(self):
        return self.X, self.y
    
    def getSplitData(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def getTensors(self):
        return self.train_dataset, self.test_dataset

    def getDataloaders(self):
        return self.train_dataloader, self.test_dataloader

    def flatten(self):
        self.X = self.X.reshape((self.batch_size, -1))
        self._datapointShapes() # Shapes are updated to the flattened one
        return self.X

    def split(self, test_size=0.25, display=False):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.y, test_size=test_size)
        if display:
            print("> X_train.shape, Y_train.shape, X_test.shape, Y_test.shape:")
            print(self.X_train.shape, self.Y_train.shape, self.X_test.shape, self.Y_test.shape)
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def tensorize(self):
        self.train_dataset = TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.Y_train).float())
        self.test_dataset = TensorDataset(torch.from_numpy(self.X_test).float(), torch.from_numpy(self.Y_test).float())
        return self.train_dataset, self.test_dataset

    def normalize(self):
        mean = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)
        normalization = transforms.Normalize(mean=mean, std=std)

        self.train_dataset.transform = transforms.Compose([normalization])
        self.test_dataset.transform = transforms.Compose([normalization])

        return self.train_dataset, self. test_dataset

    def dataload(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        return self.train_dataloader, self.test_dataloader



