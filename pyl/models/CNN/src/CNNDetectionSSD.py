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
from .datasets.ssddataset import SSDDataset
from .samplers.sampler import Sampler
from .processing import datacompose, datatransforms, targetcompose, targettransforms

class CNNDetectionSSD(CNN):
    """
    Class of method to labelize (classification + boundary boxes detection) images using SSD model.
    The model is based on the ``'SSD: Single Shot MultiBox Detector'`` paper, published in 2015.

    The weights and backward compatibility are provided by PyTorch torchvision hub.
    - SSD300 is the <300> ``version``
    - SSD320 is the <320> (lite/light) ``version`` 
    - SSD512 ins't provided but can be loaded from custom ``weights`` as the <512> ``version`` 
    """

    def __init__(self, df:pd.DataFrame, name:str=None, version="300", **kwargs) -> None:
        """
        Parameters
        ----------
        - df: the pandas dataframe (csv file) featuring the datapoints paths and its associated labels
        - name: the name (without the .pth extension) of the model to load/save under the ``/weights/`` folder
        - version: the version of the model to load
            - To load the SSD300 architecture, choose ``version="300"``
            - To load the SSD320 (MobileNet backbone) architecture, choose ``version="320"``
        """

        if name:
            self.name = name
        else:
            self.name = "default"

        if version in ["300", "320", "512"]:
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

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"SSD{self.version}_{self.name}_label_encoder.json")):
            # Tries to retreive the existing dictionnary, otherwise self.sample_batch() will generate a new one
            with open(os.path.join(WEIGHTS_WORKING_PATH, f"SSD{self.version}_{self.name}_label_encoder.json"), "r") as json_file:
                self.label_encoder = json.load(json_file)
                self.num_classes = len(self.label_encoder) + 1
        else:
            self.label_encoder = None
            self.num_classes = None
        print("DetectionSSD >> Encoded labels:", self.label_encoder, "\nDetectionSSD >> Number of classes:", self.num_classes)

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"SSD{self.version}_{self.name}.pth")):
            # If the custom weights exist, they are loaded
            self.model = torch.load(os.path.join(WEIGHTS_WORKING_PATH, f"SSD{self.version}_{self.name}.pth"))
            print("DetectionSSD >> Custom weights loaded:", f"SSD{self.version}_{self.name}.pth")

        else:
            if os.path.exists(os.path.join(WEIGHTS_LEGACY_PATH, f"SSD{self.version}.pth")):
                # If the custom wieghts do not exist, the local default pretrained version is loaded
                if self.version == "300":
                    self.model = models.detection.ssd300_vgg16()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "SSD300.pth")))
                elif self.version == "320":
                    self.model = models.detection.ssdlite320_mobilenet_v3_large()
                else:
                    self.model = models.detection.ssd300_vgg16()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "SSD300.pth")))

                print("DetectionSSD >> Default weights loaded:", f"SSD{self.version}.pth")

            else:
                # If the local default version doesn't exist, it is downloaded from PyTorch hub
                if self.version == "300":
                    self.model = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "SSD300.pth"))
                elif self.version == "320":
                    self.model = models.detection.ssdlite320_mobilenet_v3_large()
                else:
                    self.version = "300"
                    self.model = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "SSD300.pth"))

                print("DetectionSSD >> Default legacy weights dowloaded from: PyTorch hub")
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
            - "default": default image processing and SSD resizing
            - torchvision.Compose: torchvision.Compose pipeline object
            - datatransforms.Compose: custom PyYel processing.datatransforms.Compose pipeline object
        - target_transform: the labels preprocessing pipeline to load. Can be overwritten by a custom Compose object.
            - "default": default labels processing and SSD resizing
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

        size = (300, 300)
        if self.version == "300":
            size = (300, 300)
        elif self.version == "320":
            size = (320, 320)
        elif self.version == "512":
            size = (512, 512)

        if data_transform == "default" or target_transform == "default":
            self.data_transform = datacompose.DataCompose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=size)
            ])
            self.target_transform = targetcompose.TargetCompose([
                targettransforms.LabelEncode(label_encoder=self.label_encoder, column=-1),
                transforms.ToTensor(),
                targettransforms.BboxResize(x_coeff=size[0], y_coeff=size[1]),
                targettransforms.AddBackground()
            ])
        else:
            self.data_transform = data_transform
            self.target_transform = target_transform

        sampler = Sampler(df=self.df, device=self.device)

        # Sampler outputs the datapoint path, as well as the corresponding labels rows from the DB
        # In the context of SSD, i.e. object detection, the output is as follows:
        # labels_list = [(datapoint_key, class_int, x_min, y_min, x_max, y_max, class_txt), ...]
        datapoints_list, labels_list, unique_txt_classes = sampler.load_from_df(labels_type="Image_detection")

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

        labels_list = [array[:, -5:] for array in labels_list] # [..., x_min, y_min, x_max, y_max, class_txt]


        # The Sampler objects are overwritten with the list of dictionnaries
        sampler.split_in_two(datapoints_list=datapoints_list, labels_list=labels_list, test_size=test_size)
        self.train_dataloader, self.test_dataloader = sampler.send_to_dataloader(dataset=SSDDataset, 
                                                                                 data_transform=self.data_transform, target_transform=self.target_transform, 
                                                                                 **kwargs)

        return self.train_dataloader, self.test_dataloader

    def train_model(self, 
                    num_epochs:int=100,
                    bbox_retraining:bool=False,
                    backbone_retraining:bool=False,
                    lr:float=None,
                    lr_steps:list[int]=None,
                    gamma:float=0.1,
                    weight_decay:float=None
                    ):
        """
        Retrains the loaded SSD/VGG model on the previoulsly sampled batch.
        
        Parameters
        ----------
        - num_epochs: the number of epochs to train on
        - bbox_retraining: whereas to retrain the bbox regression head or not
        - backbone_retraining: whereas to retrain the backbone or not
        - lr: the learning rate. 
            If None, default values from the pretrained weights recipe is used
        - lr_steps: a list of epochs to reach to multiply the learning rate by a factor gamma.
            If None, default values from the pretrained weights recipe is used
        - gamma: the factor to multiply the learning rate by every time an epoch reaches a lr_steps milestone.
            If None, default values from the pretrained weights recipe is used
        - weight_decay: the regularisation to apply to the loss to prevent overfitting.
            If None, default values from the pretrained weights recipe is used

        Returns
        -------
        - losses_list: the list of loss value computed for every batch
        - mean_ious: the list of mean_iou computed for every batch
        """
        self._assert_model()

        # Complete retraining of the VGG backbone
        self.model.backbone.requires_grad_(backbone_retraining)
        # Boundary boxes retraining
        self.model.head.regression_head.requires_grad_(bbox_retraining)
        # The classifying head must always be retrained (no transfer learning possible)
        self.model.head.classification_head.requires_grad_(True)

        # Transfer learning
        if self.new_model:
            # The classifying head is replaced
            print("DetectionSSD >> A new head is retrained")
            if self.version == "300":
                new_cls = models.detection.ssd300_vgg16(num_classes=self.num_classes).head.classification_head
                self.model.head.classification_head = new_cls.requires_grad_(True)
            elif self.version == "320":
                new_cls = models.detection.ssdlite320_mobilenet_v3_large(num_classes=self.num_classes).head.classification_head
                self.model.head.classification_head = new_cls.requires_grad_(True)

        if not lr: # pretrained recipe lr
            if self.version == "320":
                # lr = 0.15
                lr = 0.001
            else:
                # lr = 0.002
                lr = 0.0001

        if not weight_decay: # pretrained recipe weight_decay
            if self.version == "320":
                weight_decay = 0.00004
            else:
                weight_decay = 0.0005

        if not lr_steps: # pretrained recipe lr_steps
            if self.version == "300":
                lr_steps = [80, 100]
            else:
                lr_steps = [num_epochs//2]

        if not gamma: # pretrained recipe gamma
            if self.version == "300":
                gamma = 0.2
            else:
                gamma = 0.1

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = GradScaler()

        scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=gamma)

        self.model.to(self.device)
        self.model.train()

        running_loss = 0.0
        losses_list = []
        best_loss = 1e12
        print("DetectionSSD >> Model is training on the training dataset")
        for epoch in tqdm(range(num_epochs)):
            for (images, boxes, labels) in tqdm(self.train_dataloader):

                targets = [{"boxes": box.to(self.device), "labels": label.to(self.device)} for box, label in zip(boxes, labels)]
                images = images.to(self.device)
                optimizer.zero_grad()  

                if self.device == "cuda":
                    # Mixed precision acceleration
                    with autocast():
                        output = self.model(images, targets)  
                        # In training, the model will return the bbox and the class losses
                        if bbox_retraining:
                            loss = output["classification"] + output["bbox_regression"]  
                        else:
                            loss = output["classification"] 

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    output = self.model(images, targets)  
                    # In training, the model will return the bbox and the class losses
                    if bbox_retraining:
                        loss = output["classification"] + output["bbox_regression"] 
                    else:
                        loss = output["classification"]

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                losses_list.append(loss.item())

            if loss.item() < best_loss:
                loss_epoch = epoch
                best_loss = loss.item()
                self.save_model(name=self.name)
            scheduler.step()

        # Training dataset evaluation
        self.model.eval()
        print("DetectionSSD >> Model is evaluating the training dataset")
        mean_ious = []
        with torch.no_grad():
            pred_per_image = []
            for batch_idx, (images, true_boxes, true_labels) in tqdm(enumerate(self.train_dataloader)):
                images = images.to(self.device) 
                # In eval, the model returns a dict of bbox, scores and classes
                predictions = self.model(images)

                mean_iou = 0
                for idx, prediction in enumerate(predictions):
                    pred_boxes, pred_labels, pred_scores = prediction["boxes"], prediction["labels"], prediction["scores"]
                    
                    top_boxes, top_scores, top_labels = self.filter_top_predictions(boxes=pred_boxes, labels=pred_labels, scores=pred_scores)
                    pred_per_image.append((top_boxes, top_scores, top_labels))
                    
                    mean_iou += self.detection_metrics(pred_boxes=top_boxes, pred_labels=top_labels,
                                                 true_boxes=true_boxes[idx].cpu().numpy(), true_labels=true_labels[idx].cpu().numpy())
                print("DetectionSSD >> Training mean IoU:", mean_iou/(idx+1))
                mean_ious.append(mean_iou/(idx+1))

        return losses_list, mean_ious


    def test_model(self, display=False, threshold=0.5):
        """
        Evaluates the model performance on a testing dataset that is different from the data used during training
        """
        self._assert_model()

        self.model.eval()
        print("DetectionSSD >> Model is evaluating the testing dataset")
        with torch.no_grad():
            pred_per_image = []
            for batch_idx, (images, true_boxes, true_labels) in enumerate(tqdm(self.test_dataloader)):
                images = images.to(self.device)

                predictions = self.model(images)

                for img_idx, prediction in enumerate(predictions):
                    boxes, labels, scores = prediction["boxes"].detach().cpu(), prediction["labels"].detach().cpu(), prediction["scores"].detach().cpu()
                    
                    top_boxes, top_scores, top_labels = self.filter_top_predictions(boxes=boxes, labels=labels, scores=scores, threshold=threshold)
                    pred_per_image.append((top_boxes, top_scores, top_labels))

                    iou = self.detection_metrics(pred_boxes=top_boxes.numpy(), pred_labels=top_labels.numpy(),
                                                 true_boxes=true_boxes[img_idx].cpu().numpy(), true_labels=true_labels[img_idx].cpu().numpy())

                    if display:
                        print(top_boxes, top_scores, top_labels)
                        self.plot_image_with_boxes(image=images[img_idx].detach().cpu().numpy(),
                                                    boxes=top_boxes.numpy(),
                                                    scores=top_scores.numpy(),
                                                    labels=top_labels.numpy(),
                                                    true_boxes=true_boxes[img_idx].cpu().numpy(),
                                                    true_labels=true_labels[img_idx].cpu().numpy(),
                                                    filename=f"batch{batch_idx}_img{img_idx}_iou={iou}",
                                                    threshold=threshold,
                                                    label_encoder=self.label_encoder)

        return pred_per_image

    def evaluate_datapoint(self, datapoint:Image.Image):
        """
        Evaluates an image and returns its class.
        The input should be a PIL.Image.Image object.
        Keep in mind the output is the integer representing a class as learned by the model in use,
        and thus the corresponding "text" class may vary 
        """
        self._assert_model()

        # if self.version == "300":
        #     datapoint = self.transform(input.resize((300,300))).unsqueeze(dim=0)
        # elif self.version == "320":
        #     datapoint = self.transform(input.resize((320,320))).unsqueeze(dim=0)
        # else:
        #     datapoint = self.transform(input.resize((300,300))).unsqueeze(dim=0)

        self.model.eval()
        return self.model(datapoint)
    
    def save_model(self, name:str=None):
        """
        Saves both current model and its weights into the /weights/ folder.
        Automatically called at the end of a training session.

        If no name is specified, the previous weights will be replaced by the newly computed ones.
        If this is a new model, a default weights file will be created (might delete a previously unamed model).

        Parameters
        ----------
        name: the name to save the models' weights with. Extension (.pth) should not be mentionned. 
        """
        self._assert_model()

        if name:
            self.name = name

        torch.save(self.model, os.path.join(WEIGHTS_WORKING_PATH, f"SSD{self.version}_{self.name}.pth"))
        with open(os.path.join(WEIGHTS_WORKING_PATH, f"SSD{self.version}_{self.name}_LabelEncoder.json"), "w") as json_file:
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

    model = DetectionSSD(conn=conn, name="AfricaWildlife", version="320")
    model.load_model()
    model.sample_batch(subdataset_name="AfricaWildlife", batch_size=30, test_size=0.20, num_workers=0)
    # model.train_model(num_epochs=1, backbone_retraining=False, bbox_retraining=False)
    output = model.test_model(display=True)
    # print(output[0:20])

    #model.save_model(name="ssd_test")
    
    #label = model.evaluate_datapoint(input=Image.open("C:\\Users\\nblidi\\Projets\\Datasets\\testprelab\\001\\001-1.png").convert('RGB'))
    # label = model.evaluate_datapoint(input=Image.open(":\\Users\\nblidi\\Projets\\Datasets\\testprelab\\001\\001-1.png").convert('RGB'))

    # plt.imshow(Image.open("C:\\Users\\nblidi\\Projets\\Datasets\\testprelab\\001\\001-1.png").resize((224, 224)))
    # plt.title(f"Label is: {label}")
    # plt.show()

