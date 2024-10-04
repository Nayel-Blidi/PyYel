from abc import ABC, abstractmethod
import torch

class CustomTransform(ABC):
    @abstractmethod
    def __call__(self, data):
        return data

class Transpose(CustomTransform):
    def __init__(self):
        """
        """

    def __call__(self, tensor:torch.Tensor):
        """
        Takes a 2D tensor and returns its transpose.
        """
        
        tensor = tensor.transpose(1, 2)

        return tensor