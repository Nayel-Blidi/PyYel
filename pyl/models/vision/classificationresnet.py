import os
import sys
import torch 
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from tqdm import tqdm
from PIL import Image
import json
import pandas as pd

NETWORKS_DIR_PATH = ""
if __name__ == "__main__":
    sys.path.append(os.path.dirname(NETWORKS_DIR_PATH))

WEIGHTS_WORKING_PATH = os.path.join(NETWORKS_DIR_PATH, "temp")
WEIGHTS_LEGACY_PATH = os.path.join(NETWORKS_DIR_PATH, "temp")

from ..modelsabstract import ModelsAbstract
from .datasets.resnetdataset import ResnetDataset
from ..sampler import Sampler
from .processing import datacompose, datatransforms, targetcompose, targettransforms

class ClassificationResNet(ModelsAbstract):
    """
    Class of method to labelize (classification simple/multi) images using ResNet model.
    The model is based on the ``'Deep Residual Learning for Image Recognition'`` paper, published in 2015.
    
    The weights and backward compatibility are provided by PyTorch torchvision hub.
    - ResNet18 is the ``18`` version
    - ResNet34 is the ``34`` version
    - ResNet50 is the ``50`` version
    - ResNet101 is the ``101`` version
    - ResNet152 is the ``152`` version
    """

    def __init__(self, df:pd.DataFrame, name:str=None, version="18", **kwargs) -> None:
        """
        Args
        ----
        - name: the name (without the .pth extension) of the model saved under the ``/weights/`` folder
        - version: the version of the model to load
            - To load the ResNet18 architecture, choose ``version="18"``
            - To load the ResNet34 architecture, choose ``version="34"``
            - To load the ResNet50 architecture, choose ``version="50"``
            - To load the ResNet101 architecture, choose ``version="101"``
            - To load the ResNet152 architecture, choose ``version="152"``
        """

        if name:
            self.name = name
        else:
            self.name = "default"

        if version in ["18", "34", "50", "101", "152"]:
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

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"ResNet{self.version}_{self.name}_label_encoder.json")):
            with open(os.path.join(WEIGHTS_WORKING_PATH, f"ResNet{self.version}_{self.name}_label_encoder.json"), "r") as json_file:
                self.label_encoder = json.load(json_file)
                self.num_classes = len(self.label_encoder)
        else:
            self.label_encoder = None
            self.num_classes = None 
        print("ClassificationResNet >> Encoded labels:", self.label_encoder)
        print("ClassificationResNet >> Number of classes:", self.num_classes)

        if os.path.exists(os.path.join(WEIGHTS_WORKING_PATH, f"ResNet{self.version}_{self.name}.pth")):
            # If the custom weights exist, they are loaded
            self.model = torch.load(os.path.join(WEIGHTS_WORKING_PATH, f"ResNet{self.version}_{self.name}.pth"))
            print("ClassificationResNet >> Custom weights loaded:", f"ResNet{self.version}_{self.name}.pth")

        else:
            if os.path.exists(os.path.join(WEIGHTS_LEGACY_PATH, f"ResNet{self.version}.pth")):
                # If the custom wieghts do not exist, the local default pretrained version is loaded

                if self.version == "18":
                    self.model = models.resnet18()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "ResNet18.pth")))
                elif self.version == "34":
                    self.model = models.resnet34()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "ResNet34.pth")))
                elif self.version == "50":
                    self.model = models.resnet50()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "ResNet50.pth")))
                elif self.version == "101":
                    self.model = models.resnet101()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "ResNet101.pth")))
                elif self.version == "152":
                    self.model = models.resnet152()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "ResNet152.pth")))
                else:
                    self.version = "18"
                    self.model = models.resnet18()
                    self.model.load_state_dict(torch.load(os.path.join(WEIGHTS_LEGACY_PATH, "ResNet18.pth")))

                print("ClassificationResNet >> Default legacy weights loaded:", f"ResNet{self.version}.pth")

            else:
                # If the local default version doesn't exist, it is downloaded from PyTorch

                if self.version == "18":
                    self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "ResNet18.pth"))
                elif self.version == "34":
                    self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "ResNet34.pth"))
                elif self.version == "50":
                    self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "ResNet50.pth"))
                elif self.version == "101":
                    self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "ResNet101.pth"))
                elif self.version == "152":
                    self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "ResNet152.pth"))
                else:
                    self.version = "18"
                    self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                    torch.save(self.model.state_dict(), os.path.join(WEIGHTS_LEGACY_PATH, "ResNet18.pth"))

                print("ClassificationResNet >> Default legacy weights dowloaded from: PyTorch hub")
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
            - "default": default image processing and ResNet resizing
            - torchvision.Compose: torchvision.Compose pipeline object
            - datatransforms.Compose: custom PyYel processing.datatransforms.Compose pipeline object
        - target_transform: the labels preprocessing pipeline to load. Can be overwritten by a custom Compose object.
            - "default": default labels processing and ResNet resizing
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

        self.assert_model(self.model)

        sampler = Sampler(df=self.df, device=self.device)

        # Sampler outputs the datapoint path, as well as the corresponding labels rows from the DB
        # In the context of SSD, i.e. object detection, the output is as follows:
        # labels_list = [(datapoint_key, class_int, x_min, y_min, x_max, y_max, class_txt), ...]
        datapoints_list, labels_list, unique_txt_classes = sampler.load_from_df(labels_type="Image_classification")

        datapoints_list = self.df["path"].tolist()
        labels_list = self.df["class_txt"].tolist()
        unique_txt_classes = pd.unique(labels_list)

        # If a label_encoder wasn't loaded, a new one is created
        if not self.label_encoder:
            self.label_encoder = {}
            for idx, class_txt in enumerate(unique_txt_classes):
                self.label_encoder[class_txt] = idx
            self.num_classes = len(self.label_encoder)

        if data_transform == "default" or target_transform == "default":
            self.data_transform = datacompose.DataCompose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=(224, 224))
            ])
            self.target_transform = targetcompose.TargetCompose([
                targettransforms.LabelEncode(label_encoder=self.label_encoder),
                targettransforms.OneHotEncode(num_classes=self.num_classes),
                transforms.ToTensor()
            ])
        if data_transform == "grayscale" or target_transform == "grayscale":
            self.data_transform = datacompose.DataCompose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.225]),
                transforms.Resize(size=(224, 224))
            ])
            self.target_transform = targetcompose.TargetCompose([
                targettransforms.LabelEncode(label_encoder=self.label_encoder),
                targettransforms.OneHotEncode(num_classes=self.num_classes),
                transforms.ToTensor()
            ])
        else:
            self.data_transform = data_transform
            self.target_transform = target_transform

        # The Sampler objects are overwritten with the list of dictionnaries
        sampler.split_in_two(datapoints_list=datapoints_list, labels_list=labels_list, test_size=test_size)
        self.train_dataloader, self.test_dataloader = sampler.send_to_dataloader(dataset=ResnetDataset, 
                                                                                 data_transform=self.data_transform, target_transform=self.target_transform,
                                                                                 **kwargs)

        return self.train_dataloader, self.test_dataloader

    def train_model(self, 
              num_epochs:int=10,
              lr:float=0.001,
              backbone_retraining=False,
              in_channels:int=None
              ):
        """
        Retrains the loaded ResNet model on the previoulsly sampled batch.
        
        Args
        ----
        - num_classes: the number of classes to predict
        - num_epochs: the number of epochs to train on
        - lr: the learning rate
        - backbone_retraining: whereas to unfreeze the whole model or keep the pretrained weights intact
            - False: only the classifying head is retrained (default, faster, recommended)
            - True: the whole model is retrained (slower, GPU is highly recommended)

        Returns
        -------
        - train_labels: the ground truth targets
        - train_prediction: the targets predicted by the model
        - losses_list: the list of computed loss for every batch
        """
        self.assert_model(self.model)

        if not backbone_retraining:
            # The pretrained weights are kept (frozen)
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # The pretrained weights are now trainable
            for param in self.model.parameters():
                param.requires_grad = False
            print("ClassificationResNet >> The backbone is retrained")

        if self.new_model:
            # The classifying head is reshaped and retrained
            self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, self.num_classes),
                                          nn.Softmax(dim=1))
            # self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, self.num_classes))
            print("ClassificationResNet >> A new classifying head is retrained")

        if in_channels:
            self.model.conv1 = torch.nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=self.model.conv1.kernel_size,
                                        stride=self.model.conv1.stride, padding=self.model.conv1.padding, bias=self.model.conv1.bias is not None)
            with torch.no_grad():
                self.model.conv1.weight = torch.nn.Parameter(self.model.conv1.weight.sum(dim=1, keepdim=True))
            print(f"ClassificationResNet >> ResNet input has been modified to {in_channels} channels")

        for param in self.model.fc.parameters():
            # Either way, the last layer is retrained, so a gradient is embedded in case the model actually resumes training
            param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(self.device)
        self.model.train() 
        
        running_loss = 0.0
        losses_list = []
        best_loss = 1e12
        for epoch in tqdm(range(num_epochs)):
            for inputs, targets in tqdm(self.train_dataloader):

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()  
                outputs = self.model(inputs)  

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                losses_list.append(loss.item())

            if loss.item() < best_loss:
                loss_epoch = epoch
                best_loss = loss.item()
                self.save_model(name=self.name)

        with torch.no_grad():
            for train_features, train_labels in self.train_dataloader: # Tests overfitting

                train_labels = train_labels.numpy().tolist()
                train_predicted = self.model(train_features.to(self.device)).cpu().numpy()

                # for idx, labels in tqdm(enumerate(train_predicted)):
                #     self.plot_image(image=train_features[idx].numpy(), output=labels,
                #                     labels=np.argsort(labels)[::-1], label_encoder=self.label_encoder, filename=idx)

                train_success = np.sum(np.equal(np.argmax(train_predicted, axis=1), np.argmax(train_labels, axis=1)))
                print(f"Training Acc@1 accuracy: {train_success/len(train_predicted):4f}")
    
        return train_labels, train_predicted, losses_list

    def test_model(self, output_path:str=None, display=False):
        """
        Test the real accuracy on the testing batches from the testing dataloader 

        Returns
        -------
        - test_labels: the ground truth targets
        - test_prediction: the targets predicted by the model
        """
        self.assert_model(self.model)

        self.model.eval()
        self.model.to(self.device)
        df = []
        with torch.no_grad():
            for idx, (test_features, test_labels) in enumerate(self.test_dataloader): # Tests real accuracy
                
                test_labels = test_labels.numpy().tolist()
                test_predicted = self.model(test_features.to(self.device)).cpu().numpy()
                test = list(zip(test_labels, test_predicted.tolist()))

                for label, pred in test:
                    df.append(pd.DataFrame([[np.argmax(label)] + pred], columns=["GT", *self.label_encoder.keys()]))
                
                if display:
                    self.plot_image(filename=os.path.join(output_path, f"{idx}.png"),
                                    image = test_features[-1, ...], label=np.argmax(test_predicted[-1, ...]),
                                    label_encoder=self.label_encoder) 

                test_success = np.sum(np.equal(np.argmax(test_predicted, axis=1), np.argmax(test_labels, axis=1)))
                print(f"Testing Acc@1 accuracy: {test_success/len(test_predicted):.4f}")

        df = pd.concat(df)
        df.to_csv(os.path.join(output_path, "testing_results.csv"), index=False)

        return df

    def evaluate_datapoint(self, input:Image.Image):
        """
        Evaluates an image and returns its class.
        The input should be a PIL.Image.Image object.
        Keep in mind the output is the integer representing a class as learned by the model in use,
        and thus the corresponding "text" class may vary 

        Args
        ----
        - image: PIL.Image object to evaluate.

        Eg:
        >>> evaluate_datapoint(image=PIL.Image.open("path/image.png"))
        """
        self.assert_model(self.model)

        input = self.transform(input.resize((224,224))).unsqueeze(dim=0)
        self.model.eval()
        output = self.model(input).numpy()
        return output[-1][np.argsort(output[-1])][::-1]
    
    def save_model(self, name:str=None):
        """
        Saves the current model and its weights to the /weights/ folder.
        Automatically called at the end of a training session.

        If no name is specified, the previous weights will be replaced by the newly computed ones.
        If this is a new model, a default weights file will be created (might delete a previously unamed model).

        Args
        ----
        name: the name to save the models' weights with. Extension (.pth) should not be mentionned. 
        """

        if name:
            self.name = name
        
        torch.save(self.model, os.path.join(WEIGHTS_WORKING_PATH, f"ResNet{self.version}_{self.name}.pth"))
        with open(os.path.join(WEIGHTS_WORKING_PATH, f"ResNet{self.version}_{self.name}_LabelEncoder.json"), "w") as json_file:
            json.dump(self.label_encoder, json_file)

        return None
    


if __name__ == "__main__":

    from PIL import Image
    
    model = ClassificationResNet(name="BreastCancer_backbone_v2", version="50")
    model.load_model()
    model.sample_batch(subdataset_name="BreastCancer", test_size=0.50, batch_size=100)
    # _, _, losses_list = model.train_model(num_epochs=100, backbone_retraining=True, lr=0.001)
    model.test_model()

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.semilogy(losses_list)
    # plt.show()

