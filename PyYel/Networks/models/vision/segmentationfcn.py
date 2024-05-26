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


NETWORKS_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if __name__ == "__main__":
    sys.path.append(os.path.dirname(NETWORKS_DIR_PATH))

WEIGHTS_WORKING_PATH = os.path.join(NETWORKS_DIR_PATH, "models", "torchvision", "weights")
WEIGHTS_LEGACY_PATH = os.path.join(NETWORKS_DIR_PATH, "models", "torchvision", "weights_legacy")

try:
    mp.set_start_method('spawn')
except:
    None

from networks.models.modelsabstract import ModelsAbstract
from networks.scripts.sampler import Sampler
from networks.models.vision.datasets.fcndataset import FCNDataset
from networks.models.vision.processing import datacompose, datatransforms, targetcompose, targettransforms

class SegmentationFCN(ModelsAbstract):
    """
    Class of method to segment images using FCN model.
    The model is based on the ``'Fully Convolutional Networks for Semantic Segmentation'`` paper, published in 2014.

    The weights are provided by PyTorch torchvision hub. BACKWARD COMPATIBILIY ISN'T GUARANTED.
    - FCNResNet50 is the ``ResNet50`` version
    - FCNResNet101 is the ``ResNet101`` version
    """

    def __init__(self, name:str=None, version="ResNet50", **kwargs) -> None:
        """
        Args
        ----
        - conn: a SQL (sqlite3) connection object, that links the classes to the database
        - name: the name (without the .pth extension) of the model to load/save under the ``/weights/`` folder
        - version: the version of the model
            - To load a FCNResNet50 architecture, choose ``version="ResNet50"``
            - To load a FCNResNet101 architecture, choose ``version="ResNet101"``
        """

        if name:
            self.name = name
        else:
            self.name = "default"

        if version in ["ResNet50", "ResNet101"]:
            self.version = version
        else:
            raise ValueError("Invalid model version")

        self.conn = conn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None                   # Will be replaced by the loaded/created model
        self.new_model = False              # If a model is loaded from /weights/, new_model=False
        self.label_encoder:dict = None      # If a model is loaded from /weights/, label_encoder=...label_encoder.json

    def load_model(self):
        """
        Loads the model from its weights. If the model or version doesn't exist, default weights are 
        loaded instead. If the local weights are not available, they are downloaded from PyTorch hub.
        """

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"FCN{self.version}_{self.name}_label_encoder.json")):
            # Tries to retreive the existing dictionnary, otherwise self.sample_batch() will generate a new one
            with open(os.path.join(WEIGHTS_WORKING_PATH, f"FCN{self.version}_{self.name}_label_encoder.json"), "r") as json_file:
                self.label_encoder = json.load(json_file)
                self.num_classes = len(self.label_encoder) + 1
        else:
            self.label_encoder = None
            self.num_classes = None
        print("SegmentationFCN >> Encoded labels:", self.label_encoder, "\nSegmentationFCN >> Number of classes:", self.num_classes)

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"FCN{self.version}_{self.name}.pth")):
            # If the custom weights exist, they are loaded
            self.model = torch.load(os.path.join(WEIGHTS_WORKING_PATH, f"FCN{self.version}_{self.name}.pth"))
            print("SegmentationFCN >> Custom weights loaded:", f"FCN{self.version}_{self.name}.pth")

        else:
            if os.path.exists(os.path.join(WEIGHTS_LEGACY_PATH, f"FCN{self.version}.pth")):
                # If the custom wieghts do not exist, the local default pretrained version is loaded
                if self.version == "ResNet50":
                    self.model = models.segmentation.fcn_resnet50()
                    state_dict = torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "FCNResNet50.pth"))
                    aux_classifier = [key for key in state_dict.keys() if key.startswith("aux_classifier.")]
                    for key in aux_classifier:
                        state_dict.pop(key, None)
                    self.model.load_state_dict(state_dict)

                elif self.version == "ResNet101":
                    self.model = models.segmentation.fcn_resnet101()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "FCNResNet101.pth")))
                else:
                    self.model = models.segmentation.fcn_resnet50()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "FCNResNet50.pth")))

                print("SegmentationFCN >> Default weights loaded:", f"FCN{self.version}.pth")

            else:
                # If the local default version doesn't exist, it is downloaded from PyTorch hub
                if self.version == "ResNet50":
                    self.model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "FCNResNet50.pth"))
                elif self.version == "ResNet101":
                    self.model = models.segmentation.fcn_resnet101(weights=models.segmentation.FCN_ResNet101_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "FCNResNet101.pth"))
                else:
                    self.version = "ResNet50"
                    self.model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "FCNResNet50.pth"))

                print("SegmentationFCN >> Default legacy weights dowloaded from: PyTorch hub")
            self.new_model = True

        self.model.to(self.device)
        return self.model

    def sample_batch(self, subdataset_name, data_transform="default", target_transform="default", test_size=0.25, **kwargs):
        """
        Querries the SubDataset ``<subdataset_name>`` data from the server. Relevant labels are
        automatically infered from the model in use and database architecture.

        Args
        ---
        - subdataset_name: the name of the SubDataset table to retreived the data from
        - transform: a ``torchvision.transforms.Compose`` object, that specifies the pipeline of preprocess to apply
        to the batch of data. If ``None`` is given, then the data is only converted to tensor and reshaped to fit
        the model inputs expectations. 
        - label_encoder: a sklearn.preprocessing.LabelEncoder object, that attributes an integer target to a text class.
        If ``None`` is provided, a new encoding is applied. If the data nor the model underwent no change, then the encoding will
        stay identical to the previously encoded object. Otherwise, it might differ, which will require a slightly longer training
        but shouldn't degrade the overall results when using frozen pre-trained weights.
        """

        sampler = Sampler(conn=self.conn, device=self.device)

        # Sampler outputs the datapoint path, as well as the corresponding labels rows from the DB
        # In the context of SSD, i.e. object detection, the output is as follows:
        # labels_list = [(datapoint_key, class_int, x_min, y_min, x_max, y_max, class_txt), ...]
        datapoints_list, labels_list, unique_txt_classes = sampler.load_from_db(datapoints_type="Image_datapoints",
                                                                                subdataset_name=subdataset_name,
                                                                                labels_type="Image_segmentation")
        
        if data_transform == "default" or target_transform == "default":
            self.data_transform = datacompose.DataCompose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=(224, 224))
                ])
            self.target_transform = targetcompose.TargetCompose([
                transforms.ToTensor(),
                transforms.Resize(size=(224, 224))
                ])
        elif data_transform == "15/9" or target_transform == "15/9":
            self.data_transform = datacompose.DataCompose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=(800, 1333))
                ])
            self.target_transform = targetcompose.TargetCompose([
                transforms.ToTensor(),
                transforms.Resize(size=(800, 1333))
                ])
        elif data_transform == "hd" or target_transform == "hd":
            self.data_transform = datacompose.DataCompose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=(512, 512))
                ])
            self.target_transform = targetcompose.TargetCompose([
                transforms.ToTensor(),
                transforms.Resize(size=(512, 512))
                ])
        else:
            self.data_transform = data_transform
            self.target_transform = target_transform

        # If a label_encoder wasn't loaded, a new one is created
        if not self.label_encoder:
            self.label_encoder = {}
            for idx, class_txt in enumerate(unique_txt_classes):
                self.label_encoder[class_txt] = idx
            self.num_classes = len(self.label_encoder) + 1

        labels_list = [array[0] for array in labels_list]

        # The Sampler objects are overwritten with the list of dictionnaries
        sampler.split_in_two(datapoints_list=datapoints_list, labels_list=labels_list, test_size=test_size)
        self.train_dataloader, self.test_dataloader = sampler.send_to_dataloader(dataset=FCNDataset, 
                                                                                 data_transform=self.data_transform, target_transform=self.target_transform,
                                                                                 **kwargs)

        return self.train_dataloader, self.test_dataloader


    def train_model(self, 
              num_epochs:int=10,
              lr:float=0.001,
              backbone_retraining=False,
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

        # # debug
        # for model1, model2 in zip(self.model.named_parameters(), models.segmentation.fcn_resnet50().named_parameters()):
        #     # print(model1[0], model1[1].shape, model2[0], model2[1].shape)
        #     # print(model1[0], model2[0])
        #     if model1[0] != model2[0]:
        #         print("pretrained model:", model1[0], model1[1].shape)
        #         print("torch model:", model2[0], model2[1].shape)
        #     if model1[1].shape != model2[1].shape:
        #         print("pretrained model:", model1[0], model1[1].shape)
        #         print("torch model:", model2[0], model2[1].shape)

        # The resnet backbone
        self.model.backbone.requires_grad_(backbone_retraining)
        # The classifier is always retrained
        self.model.classifier.requires_grad_(True)

        if self.new_model:
            # The classifying head is replaced
            print("SegmentationFCN >> A new head is retrained")
            if self.version == "ResNet50":
                # self.model.classifier = models.segmentation.fcn.FCNHead(channels=self.num_classes)
                self.model.classifier[-1] = torch.nn.Conv2d(512, self.num_classes, kernel_size=1)
            if self.version == "ResNet101":
                # self.model.classifier = models.segmentation.fcn.FCNHead(channels=self.num_classes)
                self.model.classifier[-1] = torch.nn.Conv2d(512, self.num_classes, kernel_size=1)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        if self.device == "cuda": scaler = GradScaler()
        self.model.to(self.device)
        self.model.train()

        running_loss = 0.0
        losses_list = []
        best_loss = 1e12
        print("SegmentationFCN >> Model is training on the training dataset")
        for epoch in tqdm(range(num_epochs)):
            for (images, masks) in tqdm(self.train_dataloader):
                    
                optimizer.zero_grad()  

                if self.device == "cuda":
                    # Mixed precision acceleration
                    with autocast():
                        output = self.model(images, masks)['out']
                        print(output.shape)
                        # In training, the model will return the head's class & bbox losses, and the RPN's class & bbox losses
                        loss = criterion(output, masks)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    output = self.model(images)['out']
                    # print(output.shape)
                    # In training, the model will return the head's class & bbox losses, and the RPN's class & bbox losses
                    loss = criterion(output, masks)

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
        print("SegmentationFCN >> Model is evaluating the training dataset")
        mean_ious = []
        with torch.no_grad():
            pred_per_image = []
            for batch_idx, (images, masks) in tqdm(enumerate(self.train_dataloader)):

                images = images.to(self.device) 
                predictions = torch.argmax(self.model(images)['out'], dim=1)

        return images, predictions, losses_list


    def test_model(self, display=False, threshold=0.5):
        """
        Evaluates the model performance on a testing dataset that is different from the data used during training
        """
        self._assert_model()

        self.model.eval()
        print("SegmentationFCN >> Model is evaluating the testing dataset")
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
            self.name = name
        
        torch.save(self.model, os.path.join(WEIGHTS_WORKING_PATH, f"FCN{self.version}_{self.name}.pth"))
        with open(os.path.join(WEIGHTS_WORKING_PATH, f"FCN{self.version}_{self.name}_LabelEncoder.json"), "w") as json_file:
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

    model = SegmentationFCN(conn=conn, name=None, version="ResNet50")
    model.load_model()
    model.sample_batch(subdataset_name="LeafDisease", batch_size=10, test_size=0.99, num_workers=0)
    images, predictions, loss = model.train_model(num_epochs=10, backbone_retraining=False, lr=10-5)
    # model.test_model(display=True)

    print(images.shape, predictions.shape)
    # print(predictions)
    for k in range(len(images)):
        plt.subplot(1, 3, 1)    
        # plt.imshow(images[0,...])
        plt.imshow(np.clip(np.transpose(images[k,...], (1, 2, 0))*255 + 1 / 2, a_min=0, a_max=1))
        plt.subplot(1, 3, 2)
        plt.imshow(predictions[k,...])
        plt.subplot(1, 3, 3)
        plt.plot(loss)
        plt.show()
