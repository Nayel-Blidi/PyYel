import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FCNDataset(Dataset):
    """
    A custom torch dataset dedicated to the FCN input format that overwrittes 
    methods to return Dataloader's objects.
    """
    def __init__(self, datapoints_list:list, labels_list:list, 
                 data_transform:transforms.Compose=None, target_transform:transforms.Compose=None):
        
        self.datapoints_list = datapoints_list
        self.labels_list = labels_list

        self.data_transform = data_transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.datapoints_list)
    
    def __getitem__(self, idx):
        """
        Loads the datapoint and its labels as arrays, and returns the tensors expected by the model
        """
        
        path = self.datapoints_list[idx] 
        image = Image.open(os.path.normpath(path)).convert('RGB')
        # The datapoints are loaded per batch, so they are augmented every epoch
        if self.data_transform:
            image = self.data_transform(image).float() / 255
        else:
            image = torch.Tensor(image).float() / 255

        path, class_txt = self.labels_list[idx] # [(path1, class_txt1), (path2, class_txt2, ...)]
        mask = (Image.open(os.path.normpath(path)).convert("L"), class_txt)
        if self.target_transform:
            mask = self.target_transform(mask).long()
        else:
            mask = torch.Tensor(mask).long()

        return image.to(self.device), mask.to(self.device)

    def _collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
        """

        images = list()
        masks = list()

        for b in batch:
            images.append(b[0])
            masks.append(b[1])

        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0).squeeze(1) # crossentropy expects 3D inputs without in_channels dimension

        return images, masks