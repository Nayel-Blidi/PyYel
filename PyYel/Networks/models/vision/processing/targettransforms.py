from abc import ABC, abstractmethod
import torch
import numpy as np

class CustomTransform(ABC):
    @abstractmethod
    def __call__(self, target):
        return target
    
__all__ = [
    "BboxResize",
    "AddBackground"
]

class AddBackground(CustomTransform):
    def __call__(self, target:torch.tensor):
        """
        Takes a target array of shape [N, 5] and returns it with its labels increased by 1 to free the
        class 0 spot for the __background__ class.

        The expected target array is of format [x_min, y_min, x_max, y_max, label] 
        """
        target[..., -1] = torch.add(1, target[..., -1])
        return target

class YoloToStandard(CustomTransform):
    def __call__(self, target:torch.tensor):
        """
        Takes a target array of shape [N, 5] and returns it with its boundary box coordinates
        converted to the STANDARD format, i.e.:
        >>> yolo_format = [x_center, y_center, width, height]
        >>> standard_format = [x_min, y_min, x_max, y_max] 

        The expected target array is of format [x_center, y_center, width, height, label] 
        """
        # TODO
        return target

class StandardToYolo(CustomTransform):
    def __call__(self, target:torch.tensor):
        """
        Takes a target array of shape [N, 5] and returns it with its boundary box coordinates
        converted to the YOLO format, i.e.:
        >>> standard_format = [x_min, y_min, x_max, y_max] 
        >>> yolo_format = [x_center, y_center, width, height]

        The expected target array is of format [x_min, y_min, x_max, y_max, label] 
        """
        # TODO
        return target
    
class BboxResize(CustomTransform):
    def __init__(self, x_coeff=1, y_coeff=1):
        """
        - y_coeff is the coefficient alongside axis 0 (top-bottom)
        - x_coeff is the coefficient alongside axis 1 (left-right)
        """

        self.x_coeff = x_coeff
        self.y_coeff = y_coeff

    def __call__(self, target):
        """
        Takes a target array of shape [N, 5] and returns it with its boundary box coordinates multiplied.

        The expected target array is of format [x_min, y_min, x_max, y_max, label] 
        """

        target[..., -5] = torch.mul(self.x_coeff, target[..., -5])
        target[..., -4] = torch.mul(self.y_coeff, target[..., -4])
        target[..., -3] = torch.mul(self.x_coeff, target[..., -3])
        target[..., -2] = torch.mul(self.y_coeff, target[..., -2])

        return target

class LabelEncode(CustomTransform):
    def __init__(self, label_encoder, column=-1):
        """
        - label_encoder is the encoding dictionnary of format {key[str]: value[int]} 
        - column is for a 2D target array the column featuring the classes to encode of format [..., label:str, ...]  
        """
        self.label_encoder = label_encoder
        self.column = column

    def __call__(self, target):
        """
        Takes a target array of shape [N, 5] and returns it with its boundary box coordinates multiplied.

        The expected target array is of format [..., label:str, ...] 
        """

        for row_idx, row in enumerate(target):
            target[row_idx, self.column] = self.label_encoder[row[self.column]]

        return target.astype(np.float32)

class OneHotEncode(CustomTransform):
    def __init__(self, num_classes, column=-1):
        """
        - num_classes is the number of classes to one-hot encode, i.e. the length of the target vector
        - column is for a 2D target array the column featuring the classes to encode of format [..., label:str, ...]  
        """
        self.num_classes = num_classes
        self.column = column

    def __call__(self, target:np.ndarray):
        """
        Takes a target array of shape [N, 5] and returns it with its boundary box coordinates multiplied.

        The expected target array is of format [label:int] 
        """

        if len(target[:, self.column]) >= 2:
            target = np.any(np.squeeze(np.eye(self.num_classes)[target.astype(int)]), axis=0).astype(int)
        else:
            target = np.expand_dims(np.squeeze(np.eye(self.num_classes)[target.astype(int)]), axis=0).astype(int) 

        return target.reshape((1, self.num_classes))
