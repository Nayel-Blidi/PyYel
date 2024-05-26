import os
import sys

import torch 
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity

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
from networks.models.vision.datasets.fasterrcnndataset import FasterRCNNDataset
from networks.models.vision.processing import datacompose, datatransforms, targetcompose, targettransforms


class DetectionFasterRCNN(ModelsAbstract):
    """
    Class of method to labelize (classification + boundary boxes detection) images using Faster R-CNN model.
    The model is based on the ``'Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks'`` paper, published in 2015.
    - The proposed Resnet50_v1 version features an updated backbone that was replaced from a VGG16 to a ResNet50 one.
    - The proposed MobileNet version has a MobileNetV3 backbone, based on the ``'Searching for MobileNetV3'`` paper, published in 2019.
    - The proposed Resnet50_v2 version has a transformer backbone, based on the ``'Benchmarking Detection Transfer Learning with Vision Transformers'`` paper, published in 2021.

    The weights and backward compatibility are provided by PyTorch torchvision hub.
    - FasterRCNNResNet50v1 is the <ResNet50v1> (default resnet) ``version``
    - FasterRCNNResNet50v2 is the <ResNet50v2> (vision transformer) ``version`` 
    - FasterRCNNMobileNet is the <MobileNet> (light) ``version`` 
    - FasterRCNNMobileNet320 is the <MobileNet320> (very light) ``version`` 
    """

    def __init__(self, name:str=None, version="ResNet50v1", **kwargs) -> None:
        """
        Args
        ----
        - name: the name (without the .pth extension) of the model to load/save under the ``/weights/`` folder
        - version: the version of the model to load
            - To load the FasterRCNNResNet50v1 architecture, choose ``version="ResNet50v1"``
            - To load the FasterRCNNResNet50v2 architecture, choose ``version="ResNet50v2"``
            - To load the FasterRCNNMobileNet architecture, choose ``version="MobileNet"``
            - To load the FasterRCNNMobileNet320 architecture, choose ``version="MobileNet320"``
        """

        if name:
            self.name = name
        else:
            self.name = "default"

        if version in ["ResNet50v1", "ResNet50v2", "MobileNet", "MobileNet320"]:
            self.version = version
        else:
            raise ValueError("DetectionFasterRCNN >> Invalid model version")

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

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"FasterRCNN{self.version}_{self.name}_LabelEncoder.json")):
            # Tries to retreive the existing dictionnary, otherwise self.sample_batch() will generate a new one
            with open(os.path.join(WEIGHTS_WORKING_PATH, f"FasterRCNN{self.version}_{self.name}_LabelEncoder..json"), "r") as json_file:
                self.label_encoder = json.load(json_file)
                self.num_classes = len(self.label_encoder) + 1
        else:
            self.label_encoder = None
            self.num_classes = None
        print("DetectionFasterRCNN >> Encoded labels:", self.label_encoder, "\nDetectionFasterRCNN >> Number of classes:", self.num_classes)

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"FasterRCNN{self.version}_{self.name}.pth")):
            # If the custom weights exist, they are loaded
            self.model = torch.load(os.path.join(WEIGHTS_WORKING_PATH, f"FasterRCNN{self.version}_{self.name}.pth"))
            print("DetectionFasterRCNN >> Custom weights loaded:", f"FasterRCNN{self.version}_{self.name}.pth")

        else:
            if os.path.exists(os.path.join(WEIGHTS_LEGACY_PATH, f"FasterRCNN{self.version}.pth")):
                # If the custom wieghts do not exist, the local default pretrained version is loaded

                if self.version == "ResNet50v1":
                    self.model = models.detection.fasterrcnn_resnet50_fpn()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNResNet50v1.pth")))
                elif self.version == "ResNet50v2":
                    self.model = models.detection.fasterrcnn_resnet50_fpn_v2()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNResNet50v2.pth")))
                elif self.version == "MobileNet":
                    self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNMobileNet.pth")))
                elif self.version == "MobileNet320":
                    self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNMobileNet320.pth")))
                else:
                    self.model = models.detection.fasterrcnn_resnet50_fpn()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNResNet50v1.pth")))

                print("DetectionFasterRCNN >> Default weights loaded from:", os.path.join(WEIGHTS_LEGACY_PATH, f"FasterRCNN{self.version}.pth"))

            else:
                # If the local default version doesn't exist, it is downloaded from PyTorch hub

                if self.version == "ResNet50v1":
                    self.model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNResNet50v1.pth"))
                elif self.version == "ResNet50v2":
                    self.model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNResNet50v2.pth"))
                elif self.version == "MobileNet":
                    self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNMobileNet.pth"))
                elif self.version == "MobileNet320":
                    self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNMobileNet320.pth"))
                else:
                    self.version = "ResNet50v1"
                    self.model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "FasterRCNNResNet50v1.pth"))

                print("DetectionFasterRCNN >> Default weights dowloaded from: PyTorch hub")
            self.new_model = True

        self.model.to(self.device)
        return self.model
    

    def sample_batch(self, subdataset_name, data_transform="15/9", target_transform="15/9", test_size=0.25, **kwargs):
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
                                                                                labels_type="Image_detection")
        
        if data_transform == "15/9" or target_transform == "15/9":
            self.data_transform = datacompose.DataCompose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=(800, 1333))]
                )
            self.target_transform = targetcompose.TargetCompose([
                targettransforms.LabelEncode(label_encoder=self.label_encoder, column=-1),
                transforms.ToTensor(),
                targettransforms.BboxResize(x_coeff=1333, y_coeff=800),
                targettransforms.AddBackground()
            ])

        elif data_transform == "1/1" or target_transform == "1/1":
            self.data_transform = datacompose.DataCompose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=(800,800))]
                )
            self.target_transform = targetcompose.TargetCompose([
                targettransforms.LabelEncode(label_encoder=self.label_encoder, column=-1),
                transforms.ToTensor(),
                targettransforms.BboxResize(x_coeff=800, y_coeff=800),
                targettransforms.AddBackground()
            ])
        else:
            self.data_transform = data_transform
            self.target_transform = target_transform

        # The labels are a list of tuples, where each tuple represents a box and its label
        # So it has to be regrouped as a list of subarrays, where one array represents all
        # the boxes shown on an image, i.e of shape [N, 4]
        # The FasterRCNN input is thus a list of [{boxes:[N, 4], labels:[N,]}, ...]
        
        # If a label_encoder wasn't loaded, a new one is created
        if not self.label_encoder:
            self.label_encoder = {}
            for idx, class_txt in enumerate(unique_txt_classes):
                self.label_encoder[class_txt] = idx
            self.num_classes = len(self.label_encoder) + 1

        labels_list = [array[:, -5:] for array in labels_list] # [..., x_min, y_min, x_max, y_max, class_txt]

        # The Sampler objects are overwritten with the list of dictionnaries
        sampler.split_in_two(datapoints_list=datapoints_list, labels_list=labels_list, test_size=test_size)
        self.train_dataloader, self.test_dataloader = sampler.send_to_dataloader(dataset=FasterRCNNDataset, 
                                                                                 data_transform=self.data_transform, target_transform=self.target_transform,
                                                                                 **kwargs)

        return self.train_dataloader, self.test_dataloader

    def train_model(self, 
              num_epochs:int=10,
              lr:float=10e-5,
              rpn_retraining=False,
              backbone_retraining=False
              ):
        """
        Retrains the loaded FasterRCNN model on the previoulsly sampled batch.
        
        Args
        ----
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

        # The resnet backbone
        self.model.backbone.requires_grad_(backbone_retraining)
        # The region proposal network, i.e. the first stage detector
        self.model.rpn.requires_grad_(rpn_retraining)
        # The classification head and its second stage detector
        self.model.roi_heads.requires_grad_(True)

        if self.new_model:
            # The classifying head is replaced
            print("DetectionFasterRCNN >> A new head is retrained")
            if self.version == "ResNet50v1":
                self.model.roi_heads = models.detection.fasterrcnn_resnet50_fpn(num_classes = self.num_classes).roi_heads
                self.model.rpn = models.detection.fasterrcnn_resnet50_fpn(num_classes = self.num_classes).rpn
            elif self.version == "ResNet50v2":
                self.model.roi_heads = models.detection.fasterrcnn_resnet50_fpn_v2(num_classes = self.num_classes).roi_heads
                self.model.rpn = models.detection.fasterrcnn_resnet50_fpn_v2(num_classes = self.num_classes).rpn
            elif self.version == "MobileNet":
                self.model.roi_heads = models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes = self.num_classes).roi_heads
                self.model.rpn = models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes = self.num_classes).rpn
            elif self.version == "MobileNet320":
                self.model.roi_heads = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(num_classes = self.num_classes).roi_heads
                self.model.rpn = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(num_classes = self.num_classes).rpn

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scaler = GradScaler()
        self.model.to(self.device)
        self.model.train()

        running_loss = 0.0
        losses_list = []
        best_loss = 1e12
        print("DetectionFasterRCNN >> Model is training on the training dataset")
        for epoch in tqdm(range(num_epochs)):
            for (images, boxes, labels) in tqdm(self.train_dataloader):

                targets = [{"boxes": box.to(self.device), "labels": label.to(self.device)} for box, label in zip(boxes, labels)]
                images = images.to(self.device)
                optimizer.zero_grad()  

                if self.device == "cuda":
                    # Mixed precision acceleration
                    with autocast():
                        output = self.model(images, targets)  
                        losses_list.append([loss.detach().cpu() for loss in output.values()])

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
                    losses_list.append([loss.detach().cpu() for loss in output.values()])

                    # In training, the model will return the head's class & bbox losses, and the RPN's class & bbox losses
                    if rpn_retraining:
                        loss = output["loss_classifier"] + output["loss_box_reg"] + output["loss_objectness"] + output["loss_rpn_box_reg"]
                    else:
                        loss = output["loss_classifier"] + output["loss_box_reg"]

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
            
            print(output)
            if loss.item() < best_loss:
                loss_epoch = epoch
                best_loss = loss.item()
                self.save_model(name=self.name)

        # Training dataset evaluation
        self.model.eval()
        print("DetectionFasterRCNN >> Model is evaluating the training dataset")
        mean_ious = []
        with torch.no_grad():
            pred_per_image = []
            for batch_idx, (images, true_boxes, true_labels) in tqdm(enumerate(self.train_dataloader)):

                images = images.to(self.device) 
                # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, use_cuda=True) as prof:
                #     with record_function("model_inference"):
                        # In eval, the model returns a dict of bbox, scores and classes
                predictions = self.model(images)

                mean_iou = 0
                for idx, prediction in enumerate(predictions):
                    pred_boxes, pred_labels, pred_scores = prediction["boxes"].cpu().numpy(), prediction["labels"].cpu().numpy(), prediction["scores"].cpu().numpy()
                    
                    top_boxes, top_scores, top_labels = self.filter_top_predictions(boxes=pred_boxes, labels=pred_labels, scores=pred_scores)
                    pred_per_image.append((top_boxes, top_scores, top_labels))
                    
                    mean_iou += self.detection_metrics(pred_boxes=top_boxes, pred_labels=top_labels,
                                                 true_boxes=true_boxes[idx].cpu().numpy(), true_labels=true_labels[idx].cpu().numpy())
                print("DetectionFasterRCNN >> Training mean IoU:", mean_iou/(idx+1))
                mean_ious.append(mean_iou/(idx+1))
        # prof.export_chrome_trace("trace.json")

        return losses_list, mean_ious


    def test_model(self, display=False, threshold=0.5):
        """
        Evaluates the model performance on a testing dataset that is different from the data used during training
        """
        self._assert_model()

        self.model.eval()
        print("DetectionFasterRCNN >> Model is evaluating the testing dataset")
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
                                                    filename=os.path.join(os.path.dirname(NETWORKS_DIR_PATH), "outputs", f"batch{batch_idx}_img{img_idx}_iou={iou}.png"),
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

        if self.version == "default":
            datapoint = self.transform(input.resize((300,300))).unsqueeze(dim=0)
        elif self.version == "lite":
            datapoint = self.transform(input.resize((320,320))).unsqueeze(dim=0)
        else:
            datapoint = self.transform(input.resize((300,300))).unsqueeze(dim=0)

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
        torch.save(self.model, os.path.join(WEIGHTS_WORKING_PATH, f"FasterRCNN{self.version}_{self.name}.pth"))
        with open(os.path.join(WEIGHTS_WORKING_PATH, f"FasterRCNN{self.version}_{self.name}_LabelEncoder.json"), "w") as json_file:
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

    model = DetectionFasterRCNN(conn=conn, name="AfricaWildlife_v1", version="ResNet50v1")
    model.load_model()
    model.sample_batch(subdataset_name="AfricaWildlife", batch_size=1, test_size=0.81)
    losses_list, _ = model.train_model(num_epochs=50, backbone_retraining=True, rpn_retraining=True, lr=10e-5)

    import matplotlib.pyplot as plt
    plt.semilogy([loss[0] for loss in losses_list], label="loss_classifier")
    plt.semilogy([loss[1] for loss in losses_list], label="loss_box_reg")
    plt.semilogy([loss[2] for loss in losses_list], label="loss_objectness")
    plt.semilogy([loss[3] for loss in losses_list], label="loss_rpn_box_reg")
    plt.legend()
    plt.show()
    model.test_model(display=True)

