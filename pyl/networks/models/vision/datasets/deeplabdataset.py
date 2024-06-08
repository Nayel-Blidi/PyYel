import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DeeplabV3Dataset(Dataset):
    """
    A custom torch dataset dedicated to the DeeplabV3 input format that overwrittes 
    methods to return Dataloader's objects.
    """
    def __init__(self, datapoints_list:list, labels_list:list, transform:transforms.Compose=None, device:str="cpu"):
        self.datapoints_list = datapoints_list
        self.labels_list = labels_list
        self.transform = transform
        self.device = device
    
    def __len__(self):
        return len(self.datapoints_list)
    
    def __getitem__(self, idx):
        """
        Loads the datapoint and its labels as arrays, and returns the tensors expected by the model
        """
        
        path = self.datapoints_list[idx] 
        image = Image.open(os.path.normpath(path)).convert('RGB')
        image = self.transform(image).float() / 255

        target = self.labels_list[idx] # (class, x_min, y_min, x_max, y_max, class_txt)
        boxes = torch.Tensor(target["boxes"]).float() * 300
        labels = torch.add(1, torch.Tensor(target["labels"])).long() # label 0 is reserved to __background__

        return image.to(self.device), boxes.to(self.device), labels.to(self.device)

    def _collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each
