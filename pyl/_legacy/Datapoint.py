
import numpy as np
import abc

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod

class DatapointInterface(ABC):
    """
    This interface is fake.
    The interface isn't implemented, as the purpose of the abtract methods may be harder to 
    understand by the user when calling it.
    For developpers, these methods serve as guidelines to implement classes with similar functionnalities.
    """
    @abstractmethod
    def getOriginalDatapoint(self):
        pass

    @abstractmethod
    def getModifiedDatapoint(self):
        pass
    

class Datapoint():
    """
    The standard datatype used in PyYel.
    """
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


class YelDatapoint():
    """
    A standard PyYel datapoint object, with improved compatibility.

    Args:
        data: input data
    
    Methods:
        getOriginalData(): Returns the data originally fed into the class when initialized. Allows data recovery.
        getModifiedData(): Returns the data in its current state.
        resetData(): Resets the modified data to its original state. Allows reimplementation of methods.
        reshape(): Reshapes the modified data. Replaces modified data with its new reshaped format.  
    """

    def __init__(self, data) -> None:
        self.data_original = data
        self.data_modified = data
    
    def runPipeline(self):
        """
        A standard pipeline that tries to process the data into a valid format.
        If it fails, steps will have to be implemented manually. 
        """
        pipeline = [
            self.reshape(),
        ]
        [step for step in tqdm(pipeline)]
        return self.getModifiedData


    def getOriginalData(self):
        return self.data_original
    def getModifiedData(self):
        return self.data_modified
    
    def resetData(self):
        self.data_modified = self.data_original

    def reshape(self, shape=None):
        """
        Reshapes the YelDatapoint to a desired shape. 
        Args:
            shape: Desired shape.
                [None]: Automatically tries to reshape the array to a flat (batch_size, -1) 
                        or an image (batch_size, height, width, channels) format. 
                [(x1, x2, ... xn)]: Reshapes the array according to the tupple of wanted format.
        """
        if shape is None:
            self._reshape()
        else:
            try: self.data_modified.reshape(shape)
            except:
                message = f"Incompatible current shapes {self.data_modified.shape} and expected shapes {shape}." 
                raise ValueError(message)
        
    def oneHotEncode(self):
        """
        Applies one-hot encoding to a 0D or 1D input.
        Unlikely to be usefull if not applied to a targets/labels vector. 
        """
        num_classes = np.max(self.data_original) + 1
        self.data_modified = np.eye(num_classes)[self.data_modified]

    def _reshape(self):
        dims = self.data_modified.shape
        if len(dims) == 1:
            # Promotes vector to column array (1D table)
            self.data_modified = self.data_modified.reshape((-1, 1))
        elif len(dims) == 2:
            # Assumes the data is a 2D table
            None
        elif len(dims) == 3:
            if (dims[2] == 1) or (dims[2] == 3):
                # Adds batch dimension to a presumed unique image
                self.data_modified = self.data_modified.reshape((1, dims[0], dims[1], dims[2]))
            elif (dims[0] == 1) or (dims[0] == 3):
                # Changes the presumed shape (channels, height, width) to (height, width, channels)
                self.data_modified = np.transpose(self.data_modified, (1, 2, 0))
                # Adds batch dimension to a presumed unique image
                self.data_modified = self.data_modified.reshape((1, dims[0], dims[1], dims[2]))
        elif len(dims) == 4:
            if (dims[0] == 1) or (dims[0] == 3):
                # Changes the presumed shape (batch, channels, height, width) to (batch, height, width, channels)
                self.data_modified = np.transpose(self.data_modified, (1, 2, 0))
        else:
            # n-dims array, flattened to (batch, features)
            self.data_modified = self.data_modified.reshape((dims[0], -1))



class YelDataset():
    """
    A standard PyYel dataset object, of combined features and targets datapoints.

    Args:
        X: an array-like input of features
            [list]: if the input is a list of datapoints, it will be concatenated into a single array.
            [array]: if the input is an array, the first dimension will be considered to be the batch.
        Y: an array-like input of targets
            [list]: if the input is a list of targets, it will be concatenated into a single array.
            [array]: if the input is a 1D or 2D array, every line is considered to be a target of a feature.
    """

    def __init__(self, X, Y) -> None:
        self.X_original = X
        self.Y_original = Y

        self.X_modified = X
        self.Y_modified = Y


        self._ensureCompatibility()

    def getOriginalDataset(self) -> tuple:
        return self.X_original, self.Y_original

    def getModifiedDataset(self) -> tuple:
        return self.X_modified, self.Y_modified
    
    def getDataset(self) -> tuple:
        return self.X_modified, self.Y_modified
    
    def resetDataset(self):
        """
        Returns the modified X and Y data to their original input. 
        Usefull to cancel a processing step, or to reuse the datapoint without defining it again.
        """
        self.X_modified = self.X_original
        self.Y_modified = self.Y_original

    def stackDataset(self):
        """
        For an array-like of unbatched data, stacks the X and Y datapoints into a single dataset.
        If the first axis of the features is of dimension 1, it will be considered to be the batch axis. (1, :, :, ...)
        If the features are of dimension >=4, the 1st dimension will be considered to be the batch axis. (n, :, :, ...)
        If the features are of dimension <=3, the batch dimension will be added as axis=0. (m, n, :)
        """
        if type(self.X_modified) is list:
            print(self.X_modified[0].shape)
            # self.X_modified = np.stack(self.X_modified, axis=0)

            if len(self.X_modified[0].shape) >=4:
                # Assumes the passed data 1st dimension is the batch
                self.X_modified = np.concatenate(self.X_modified, axis=0)
            elif self.X_modified[0].shape[0] == 1:
                self.X_modified = np.concatenate(self.X_modified, axis=0)
            else:
                # Assumes the data is missing a batch size (most likely a list of images)
                self.X_modified = [np.expand_dims(feature, axis=0) for feature in self.X_modified]
                print(self.X_modified[0].shape)
                self.X_modified = np.stack(self.X_modified, axis=1).squeeze(axis=0)

            # if self.X_modified[0].shape[0] == 1:
            #     # The 1st dimension is the batch
            #     self.X_modified = np.stack(self.X_modified, axis=1).squeeze(axis=0)
            # else:
            #     # The 1st dimension isn't the bach, so it is added
            #     self.X_modified = [np.expand_dims(feature, axis=0) for feature in self.X_modified]
            #     print(self.X_modified[0].shape)
            #     self.X_modified = np.stack(self.X_modified, axis=1).squeeze(axis=0)

    def splitDataset(self, test_size=0.25, display=False):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_modified, self.Y_modified, test_size=test_size)
        if display:
            print("> X_train.shape, Y_train.shape, X_test.shape, Y_test.shape:")
            print(self.X_train.shape, self.Y_train.shape, self.X_test.shape, self.Y_test.shape)
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def _ensureCompatibility(self):

        if type(self.X_modified) is list:
            return True
        else:
            if self.X_modified.shape[0] == self.Y_modified.shape[0]:
                return True
            for dimensions in self.X_modified.shape:
                if dimensions in self.Y_modified.shape:
                    return True
                
        message = f"No compatible shapes found between X {self.X_original.shape} and Y {self.Y_original.shape} inputs."
        raise ValueError(message)



class Datatensor():
    """
    Tensorized datapoint. 
    Pytorch format. 
    """

    def __init__(self, X, Y) -> None:
        self.X_original = X
        self.Y_original = Y

        self.X_modified = X
        self.Y_modified = Y

        self._datapointShapes()

    def getOriginalTensors(self):
        return self.X_original, self.Y_original

    def getModifiedTensors(self):
        return self.X_modified, self.Y_modified
        
    def runPipeline(self, **kwargs):
        datapoint_pipeline = [
            self.split(**kwargs),
            self.tensorize(**kwargs),
            self.normalize(**kwargs),
            self.dataload(),
        ]
        for step in datapoint_pipeline:
            step
        return None
    
    def getSplitData(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def getTensors(self):
        return self.train_dataset, self.test_dataset

    def getDataloaders(self):
        return self.train_dataloader, self.test_dataloader
    
    def flatten(self):
        self.X_modified = self.X_modified.reshape((self.batch_size, -1))
        return self.X_modified

    def split(self, test_size=0.25, display=False, **kwargs):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X_modified, self.Y_modified, test_size=test_size)
        if display:
            print("> X_train.shape, Y_train.shape, X_test.shape, Y_test.shape:")
            print(self.X_train.shape, self.Y_train.shape, self.X_test.shape, self.Y_test.shape)
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def splitOverwrite(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def tensorize(self, dtype='float', **kwargs):
        if dtype == 'float':
            self.train_dataset = TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.Y_train).float())
            self.test_dataset = TensorDataset(torch.from_numpy(self.X_test).float(), torch.from_numpy(self.Y_test).float())
        elif dtype == 'long':
            self.train_dataset = TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.Y_train).long())
            self.test_dataset = TensorDataset(torch.from_numpy(self.X_test).float(), torch.from_numpy(self.Y_test).long())

        return self.train_dataset, self.test_dataset
        

    def normalize(self, delta=1e-6, **kwargs):
        """
        Normalizes the X features, y stays constant.
        Args:
            delta: small value to avoid division by zero error when normalizing null rows. 
        """

        mean = np.mean(self.X_modified, axis=0) + delta
        std = np.std(self.X_modified, axis=0) + delta
        normalization = transforms.Normalize(mean=mean, std=std)

        if (self.height != 1) and (self.width != 1):
            self.train_dataset = [(normalization(sample[0]), sample[1]) for sample in self.train_dataset]
            self.test_dataset = [(normalization(sample[0]), sample[1]) for sample in self.test_dataset]
        else:
            self.train_dataset = [(F.normalize(sample[0].view(1, -1)), sample[1]) for sample in self.train_dataset]
            self.test_dataset = [(F.normalize(sample[0].view(1, -1)), sample[1]) for sample in self.test_dataset]

        return self.train_dataset, self. test_dataset

    def dataload(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        return self.train_dataloader, self.test_dataloader


    def _datapointShapes(self, display=False) -> tuple:
        if len(self.X_modified.shape) == 1:
            # Data is supposed to be a column of values
            self.batch_size = self.X_modified.shape[0]
            self.in_channels = 1
            self.height = 1
            self.width = 1
        elif len(self.X_modified.shape) == 2:
            # Data is supposed to be a structured frame
            self.batch_size = self.X_modified.shape[0]
            self.in_channels = self.X_modified.shape[1]
            self.height = 1
            self.width = 1
        elif len(self.X_modified.shape) == 3:
            # Data is supposed to be a gray scale image, or layers of structured data like
            self.batch_size = self.X_modified.shape[0]
            self.in_channels = 1
            self.height = self.X_modified.shape[1]
            self.width = self.X_modified.shape[2]
            self.X_modified = np.expand_dims(self.X_modified, axis=1)
        elif len(self.X_modified.shape) == 4:
            # Data is supposed to be rgb images like
            self.batch_size = self.X_modified.shape[0]
            self.in_channels = self.X_modified.shape[1]
            self.height = self.X_modified.shape[2]
            self.width = self.X_modified.shape[3]

        self.output_size = self.Y_modified.shape[-1]
        
        if display:
            print("> batch_size, in_channels, height, width, output_size:")
            print("\t", self.batch_size, "\t" , self.in_channels, "\t", self.height, "\t", self.width, "\t", self.output_size)

        return self.batch_size, self.in_channels, self.height, self.width, self.output_size
            