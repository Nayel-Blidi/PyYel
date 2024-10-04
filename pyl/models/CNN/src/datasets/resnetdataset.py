import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json

class ResnetDataset(Dataset):
    """
    A custom torch dataset dedicated to the ResNet input format that overwrittes 
    methods to return Dataloader's objects.
    """
    def __init__(self, 
                 datapoints_list:list, labels_list:list, 
                 data_transform:transforms.Compose=None, target_transform:transforms.Compose=None):
                
        self.datapoints_list = datapoints_list
        self.labels_list = labels_list

        self.data_transform = data_transform
        self.target_transform = target_transform

        # The batching/collate function used will be the default one
        # TODO: add one for multi classification ? for multi rows 1-hot encoded target tensor
        self._collate_fn = None
    
        # The targets remain loaded, so they are augmented only once
        # In the case of segmentation for instance, the mask (which is an image) would likely be 
        # loaded during the __getitem__ process due to memory limitations
        # if self.target_transform:
        #     self.labels_list = [self.target_transform(target) for target in self.labels_list]
        # else:
        #     self.labels_list = [torch.Tensor(target) for target in self.labels_list]

    def __len__(self):
        return len(self.datapoints_list)

    def __getitem__(self, idx):
        
        path = self.datapoints_list[idx] 
        image = Image.open(os.path.normpath(path)).convert('RGB')
        # The datapoints are loaded per batch, so they are augmented every epoch
        if self.data_transform:
            image = self.data_transform(image).float() / 255
        else:
            image = torch.Tensor(image).float() / 255

        path = self.labels_list[idx]
        target = self._parse_classif_json(label_path=self.labels_list[idx])
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


    def _parse_classif_json(self, label_path: str):

        with open(label_path, 'r') as file:
            label_json = json.load(file)

        labels = []
        for label in label_json.values():
            labels.append(label["cls"])

        return labels

