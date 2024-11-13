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
    

class ToTensor(CustomTransform):
    def __init__(self, dtype=None):
        """
        Transforms a regular array into a tensor of same dtype

        Parameters
        ----------
        - dtype: the tensor dtype to transform the array into. If None, dtype is infered from array dtype
        """
        self.dtype = dtype

    def __call__(self, target):
        """
        Takes an array and converts it into a ``torch.Tensor``
        """

        if self.dtype:
            target = torch.tensor(data=target, dtype=self.dtype)
        else:
            target = torch.from_numpy(target)

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


class BboxTranspose(CustomTransform):
    def __init__(self):
        """
        - order: the order to exchange the columns into.  

        >>> target = [x_min, y_min, x_max, y_max, label]
        >>> target = BboxTranspose()(target)
        >>> target = [x_max, y_max, x_min, y_min, label] 
        """

    def __call__(self, target):
        """
        Takes a target array of shape [N, 5] and returns it with its boundary box transposed.

        The expected target array is of format [x_min, y_min, x_max, y_max, label] 
        """
        
        for idx, box in enumerate(target[0]):
            # [y_min, x_min, x_min, y_min, label] = [x_min, y_min, x_max, y_max, label]
            target[..., idx, -5] = box[-4]
            target[..., idx, -4] = box[-5]
            target[..., idx, -3] = box[-2]
            target[..., idx, -2] = box[-3]

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

        for idx, label in enumerate(target):
            target[idx] = int(self.label_encoder[label])

        return target


class MaskEncode(CustomTransform):
    def __init__(self, label_encoder, column=-1):
        """
        - label_encoder is the dictionnary to use to assign the mask pixel values 
        - label is the class_txt (str) to encode into a pixel value (int)
        - column is for a 2D target array the column featuring the classes to encode of format [..., label:str, ...]  
        """
        self.label_encoder = label_encoder
        self.column = column

    def __call__(self, target:np.ndarray):
        """
        Takes a segmentation target (mask, class_txt) and returns the encoded mask 
        """
        
        mask, class_txt = target
        mask[mask != 0] = self.label_encoder[class_txt]

        return mask

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

        # Create a one-hot encoded matrix
        one_hot_matrix = np.zeros((self.num_classes), dtype=int)
        one_hot_matrix[target] = 1

        # if len(target[:, self.column]) >= 2:
        #     target = np.any(np.squeeze(np.eye(self.num_classes)[target.astype(int)]), axis=0).astype(int)
        # else:
        #     target = np.expand_dims(np.squeeze(np.eye(self.num_classes)[target.astype(int)]), axis=0).astype(int) 

        return one_hot_matrix
