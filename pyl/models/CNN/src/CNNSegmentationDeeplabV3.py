import os
import sys

import torch 
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import MultiStepLR

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sqlite3
from tqdm import tqdm
from PIL import Image
import json
import cv2
import multiprocessing as mp
import pandas as pd

NETWORKS_DIR_PATH = ""
if __name__ == "__main__":
    sys.path.append(os.path.dirname(NETWORKS_DIR_PATH))

WEIGHTS_WORKING_PATH = os.path.join(NETWORKS_DIR_PATH, "temp")
WEIGHTS_LEGACY_PATH = os.path.join(NETWORKS_DIR_PATH, "temp")

try:
    mp.set_start_method('spawn')
except:
    None

from .CNN import CNN
from .samplers.sampler import Sampler
from .datasets.deeplabdataset import DeeplabV3Dataset

class CNNSegmentationDeeplabV3(CNN):
    """
    Class of method to labelize (segmentation) images using Deeplab V3 model.
    The model is based on the ``'Rethinking Atrous Convolution for Semantic Image Segmentation'`` paper, published in 2017.

    The weights are provided by PyTorch torchvision hub. BACKWARD COMPATIBILIY ISN'T GUARANTED.
    - DeeplabV3_ResNet50 is the <ResNet50> ``version``
    - DeeplabV3_MobileNet is the <MobileNet> (lite/light) ``version`` 
    """

    def __init__(self, df:pd.DataFrame, name:str=None, version="ResNet50", **kwargs) -> None:
        """
        Args
        ----
        - df: the pandas dataframe (csv file) featuring the datapoints paths and its associated labels
        - name: the name (without the .pth extension) of the model to load/save under the ``/weights/`` folder
        - version: the version of the model to load
        """

        self.name = name
        if version in ["ResNet50", "MobileNet"]:
            self.version = version
        else:
            raise ValueError("Invalid model version")

        self.df = df
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None                   # Will be replaced by the loaded/created model
        self.new_model = False              # If a model is loaded from /weights/, new_model=False
        self.label_encoder:dict = None      # If a model is loaded from /weights/, label_encoder=...label_encoder.json

    def load_model(self):
        """
        Loads the model from its weights. If the model or version doesn't exist, default weights are 
        loaded instead. If the local weights are not available, they are downloaded from PyTorch hub.
        """

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"DeeplabV3{self.version}_{self.name}_label_encoder.json")):
            # Tries to retreive the existing dictionnary, otherwise self.sample_batch() will generate a new one
            with open(os.path.join(WEIGHTS_WORKING_PATH, f"DeeplabV3{self.version}_{self.name}_label_encoder.json"), "r") as json_file:
                self.label_encoder = json.load(json_file)
                self.num_classes = len(self.label_encoder) + 1
        else:
            self.label_encoder = None
            self.num_classes = None
        print("SegmentationDeeplabV3 >> Encoded labels:", self.label_encoder, "\nSegmentationDeeplabV3 >> Number of classes:", self.num_classes)

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"DeeplabV3_{self.version}_{self.name}.pth")):
            # If the custom weights exist, they are loaded
            self.model = torch.load(os.path.join(WEIGHTS_WORKING_PATH, f"DeeplabV3_{self.version}_{self.name}.pth"))
            print("SegmentationDeeplabV3 >> Custom weights loaded:", f"DeeplabV3_{self.version}_{self.name}.pth")

        else:
            if os.path.exists(os.path.join(WEIGHTS_LEGACY_PATH, f"DeeplabV3_{self.version}.pth")):
                # If the custom wieghts do not exist, the local default pretrained version is loaded
                if self.version == "ResNet50":
                    self.model = models.segmentation.deeplabv3_resnet50()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "DeeplabV3_ResNet50.pth")))
                elif self.version == "MobileNet":
                    self.model = models.segmentation.deeplabv3_mobilenet_v3_large()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "DeeplabV3_MobileNet.pth")))
                else:
                    self.model = models.segmentation.deeplabv3_resnet50()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "DeeplabV3_ResNet50.pth")))

                print("SegmentationDeeplabV3 >> Default weights loaded:", f"DeeplabV3_{self.version}.pth")

            else:
                # If the local default version doesn't exist, it is downloaded from PyTorch hub
                if self.version == "ResNet":
                    self.model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "DeeplabV3_ResNet50.pth"))
                elif self.version == "MobileNet":
                    self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "DeeplabV3_MobileNet.pth"))
                else:
                    self.version = "ResNet50"
                    self.model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "DeeplabV3_ResNet50.pth"))

                print("SegmentationDeeplabV3 >> Default legacy weights dowloaded from: PyTorch hub")
            self.new_model = True

        self.model.to(self.device)
        return self.model

    def sample_batch(self, data_transform="default", target_transform="default", test_size=0.25, **kwargs):
        """
        Querries the datapoints paths and labels from the dataframe. Relevant labels are
        automatically infered from the model in use and dataframe structure.

        Args
        ---
        - data_transform: the datapoints preprocessing pipeline to load. Can be overwritten by a custom Compose object.
            - TODO
            - torchvision.Compose: torchvision.Compose pipeline object
            - datatransforms.Compose: custom PyYel processing.datatransforms.Compose pipeline object
        - target_transform: the labels preprocessing pipeline to load. Can be overwritten by a custom Compose object.
            - TODO
            - torchvision.Compose: torchvision.Compose pipeline object (not recommended)
            - targettransforms.Compose: custom PyYel processing.datatransforms.Compose pipeline object
        - test_size: the proportion of examples to allocate to the testing dataloader. Must be a value between 0 and 1.

        Kwargs
        ------
        - chunks: int = 1,
        - batch_size: int = None,
        - drop_last: bool = True,
        - num_workers: int = 0
        """

        if transform:
            # Overwrites the default infered transform
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=(800,800))]
                )

        sampler = Sampler(df=self.df, device=self.device)

        # Sampler outputs the datapoint path, as well as the corresponding labels rows from the DB
        # In the context of SSD, i.e. object detection, the output is as follows:
        # labels_list = [(datapoint_key, class_int, x_min, y_min, x_max, y_max, class_txt), ...]
        datapoints_list, labels_list, unique_txt_classes = sampler.load_from_df(datapoints_type="Image_datapoints",
                                                                                labels_type="Image_detection")

        # The labels are a list of tuples, where each tuple represents a box and its label
        # So it has to be regrouped as a list of subarrays, where one array represents all
        # the boxes shown on an image, i.e of shape [N, 4]
        # The SSD input is thus a list of [{boxes:[N, 4], labels:[N,]}, ...]
        # labels_list = np.array(labels_list, dtype=object)
        
        # If a label_encoder wasn't loaded, a new one is created
        if not self.label_encoder:
            self.label_encoder = {}
            for idx, class_txt in enumerate(unique_txt_classes):
                self.label_encoder[class_txt] = idx
            self.num_classes = len(self.label_encoder) + 1

        # The label dictionnary is reshaped as a valid SSD input, which is a dictionary 
        labels_list = [{'boxes':label_array[:, -5:-1].astype(np.float32), 'labels':[self.label_encoder[class_txt] for class_txt in label_array[:, -1]]} for label_array in labels_list]

        # The Sampler objects are overwritten with the list of dictionnaries
        sampler.split_in_two(datapoints_list=datapoints_list, labels_list=labels_list, test_size=test_size)
        self.train_dataloader, self.test_dataloader = sampler.send_to_dataloader(dataset=DeeplabV3Dataset, transform=self.transform, **kwargs)

        return self.train_dataloader, self.test_dataloader


    def train_model(self, 
              num_epochs:int=10,
              lr:float=0.001,
              rpn_retraining=False,
              backbone_retraining=False
              ):
        """
        Retrains the loaded SSD/VGG model on the previoulsly sampled batch.
        
        Args
        ----
        - num_classes: the number of classes to predict
        - num_epochs; the number of epochs to train on
        - lr: the learning rate
        - full_retraining:
            - False: only the classifying head is retrained (default, faster, recommended)
            - True: the whole model is retrained (slower, GPU is highly recommended)

        Returns
        -------
        - train_labels: the ground truth targets
        - train_prediction: the targets predicted by the model
        """

        self._assert_model()

        # The resnet backbone
        self.model.backbone.requires_grad_(backbone_retraining)
        # The region proposal network, i.e. the first stage detector
        self.model.rpn.requires_grad_(rpn_retraining)
        # The classification head and its second stage detector
        self.model.roi_heads.requires_grad_(True)

        if self.new_model:
            # The classifying head is replaced
            print("SegmentationDeeplabV3 >> A new head is retrained")
            if self.version == "ResNet50_v1":
                self.model.roi_heads = models.detection.fasterrcnn_resnet50_fpn(num_classes = self.num_classes+1).roi_heads
            if self.version == "ResNet50_v2":
                self.model.roi_heads = models.detection.fasterrcnn_resnet50_fpn_v2(num_classes = self.num_classes+1).roi_heads
            if self.version == "MobileNet":
                self.model.roi_heads = models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes = self.num_classes+1).roi_heads
            if self.version == "MobileNet_320":
                self.model.roi_heads = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(num_classes = self.num_classes+1).roi_heads

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scaler = GradScaler()
        self.model.to(self.device)
        self.model.train()

        running_loss = 0.0
        losses_list = []
        best_loss = 1e12
        print("SegmentationDeeplabV3 >> Model is training on the training dataset")
        for epoch in tqdm(range(num_epochs)):
            for (images, boxes, labels) in tqdm(self.train_dataloader):

                targets = [{"boxes": box, "labels": label} for box, label in zip(boxes, labels)]
                optimizer.zero_grad()  

                if self.device == "cuda":
                    # Mixed precision acceleration
                    with autocast():
                        output = self.model(images, targets)  
                        # In training, the model will return the head's class & bbox losses, and the RPN's class & bbox losses
                        if rpn_retraining:
                            loss = output["loss_classifier"] + output["loss_box_reg"] + output["loss_objectness"] + output["loss_rpn_box_reg"]
                        else:
                            loss = output["loss_classifier"] + output["loss_box_reg"]

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    output = self.model(images, targets)  
                    # In training, the model will return the head's class & bbox losses, and the RPN's class & bbox losses
                    if rpn_retraining:
                        loss = output["loss_classifier"] + output["loss_box_reg"] + output["loss_objectness"] + output["loss_rpn_box_reg"]
                    else:
                        loss = output["loss_classifier"] + output["loss_box_reg"]

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                losses_list.append(loss.item())

            if loss.item() < best_loss:
                loss_epoch = epoch
                best_loss = loss.item()
                self.save_model(name=self.name)

        # Training dataset evaluation
        self.model.eval()
        print("SegmentationDeeplabV3 >> Model is evaluating the training dataset")
        mean_ious = []
        with torch.no_grad():
            pred_per_image = []
            for batch_idx, (images, true_boxes, true_labels) in tqdm(enumerate(self.train_dataloader)):

                images = images.to(self.device) 
                predictions = self.model(images)

        return losses_list


    def test_model(self, display=False, threshold=0.5):
        """
        Evaluates the model performance on a testing dataset that is different from the data used during training
        """
        self._assert_model()

        self.model.eval()
        print("SegmentationDeeplabV3 >> Model is evaluating the testing dataset")
        with torch.no_grad():
            pred_per_image = []
            for batch_idx, (images, true_boxes, true_labels) in enumerate(tqdm(self.test_dataloader)):

                images = images.to(self.device)
                predictions = self.model(images)
                    
        return pred_per_image

    def evaluate_datapoint(self, datapoint:Image.Image):
        """
        Evaluates an image and returns its class.
        The input should be a PIL.Image.Image object.
        Keep in mind the output is the integer representing a class as learned by the model in use,
        and thus the corresponding "text" class may vary 
        """
        self._assert_model()

        self.model.eval()
        return self.model(datapoint)
    
    def save_model(self, name:str=None):
        """
        Saves both current model and its weights into the /weights/ folder.
        Automatically called at the end of a training session.

        If no name is specified, the previous weights will be replaced by the newly computed ones.
        If this is a new model, a default weights file will be created (might delete a previously unamed model).

        Args
        ----
        name: the name to save the models' weights with. Extension (.pth) should not be mentionned. 
        """
        self._assert_model()

        if name:
            torch.save(self.model, os.path.join(WEIGHTS_WORKING_PATH, f"{name}.pth"))
            with open(os.path.join(WEIGHTS_WORKING_PATH, f"{name}_label_encoder.json"), "w") as json_file:
                json.dump(self.label_encoder, json_file)
        else:
            torch.save(self.model, f"{self.name}.pth")
            with open(f"{self.name}_label_encoder.json", "w") as json_file:
                json.dump(self.label_encoder, json_file)
        return None

    def _assert_model(self):
        """
        Asserts a deep learning model has been loaded 
        """
        if self.model is None:
            raise ValueError("No model was loaded, so this function can't be executed")
        return True




if __name__ == "__main__":
    from database.scripts.connection import ConnectionSQLite
    from PIL import Image

    test_path = os.path.join(os.path.dirname(os.path.dirname(NETWORKS_DIR_PATH)), "Datasets", "AfricaWildlife")

    coo = ConnectionSQLite()
    conn = coo.connect_database()

    model = SegmentationDeeplabV3(conn=conn, name=None, version="ResNet50")
    model.load_model()
    model.sample_batch(subdataset_name="AfricaWildlife", batch_size=10, test_size=0.90, num_workers=0)
    model.train_model(num_epochs=1, backbone_retraining=False, rpn_retraining=False)
    model.test_model(display=True)
