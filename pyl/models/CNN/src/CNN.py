from abc import ABC, abstractmethod
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from sklearn.metrics import average_precision_score, precision_recall_curve
from datetime import datetime
import json
from PIL import Image
import shutil

from .samplers.sampler import Sampler

class CNN(ABC):
    """
    Base CNN class
    """
    def __init__(self, model_name: str, weights_path: str = None) -> None:
        """
        Inits a CNN base class, and loads it from a local checkpoint or from PyTorch Hub using 
        the Torchvision API.

        Parameters
        ----------
        model_name: str
            The name of the model to use. The folder where the weights will saved will have the same name.
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Note
        ----
        - Models are directly available at PyTorch Hub and do not require any credentials as of today. 
        """
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        if weights_path is None: weights_path = os.getcwd()
        self.model_folder = os.path.join(weights_path, model_name.split(sep='/')[-1])

        if not os.path.exists(self.model_folder):
            self._init_model()

        return None


    def _init_model(self):
        """
        Initializes a model locally by creating a recipient weights folder.
        """

        os.mkdir(path=self.model_folder)
        self._add_gitignore(folder_path=self.model_folder)

        return True

    def _repair_model(self):
        """
        Repairs a model by deleting the currrent model's snapshot and 'downloading' a new one.
        """

        self._empty_folder(self.model_folder)
        os.rmdir(self.model_folder)
        self._init_model()

        return True

    def _add_gitignore(self, folder_path: str):
        """
        Creates a local .gitignore file to ignore the weights and model's content.
        """
        gitignore_content = "*"
        gitignore_path = os.path.join(folder_path, '.gitignore')
        with open(gitignore_path, 'w') as gitignore_file:
            gitignore_file.write(gitignore_content)


    def _empty_folder(self, path: str = None):
        """
        Empties a folder.

        Parameters
        ----------
        path: str
            The path to the folder to empty from all its content
        """
        
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    # Check if it is a file and delete it
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    # Check if it is a directory and delete it and its contents
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"CNN >> Failed to empty {path}: {e}")

        return True

    @abstractmethod
    def load_model(self, epoch: int = -1):
        self.current_epoch: int
        self.state_dict: dict[int, dict[str, None]] 
        pass

    @abstractmethod
    def sample_batch(self, sampler: Sampler, batch_name: str, test_size: float = 0.25):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass

    @abstractmethod
    def evaluate_datapoint(self, images: list[Image.Image]):
        pass

    @abstractmethod
    def save_model(self):
        pass


    def save_model(self, optimizer_state:dict, model:torch.nn.Module, epoch: int = 0,
                   loss_history:list[float]=[], loss_checkpoints:float=None, 
                   name:str=None, **kwargs):
        """
        Saves the current model and its weights to the /weights/ folder.
        Automatically called at the end of a training session.

        If no name is specified, the previous weights will be replaced by the newly computed ones.
        If this is a new model, a default weights file will be created (might delete a previously unamed model).

        Parameters
        ----------
        name: the name to save the models' weights with. Extension (.pth) should not be mentionned. 
        """
        
        # Current model in use
        current_state = {
            "model": model,
            "optimizer": optimizer_state,
            "loss_history": loss_history,
            "loss_checkpoint": loss_checkpoints,
            "date": datetime.now().strftime("%Y/%m/%d:%Hh%Mm%Ss")
        }
        # last_epoch = sorted(int(epoch) for epoch in list(self.state_dict.keys()))[-1] # Last model was saved at epoch last_epoch
        # self.state_dict[f"{last_epoch+epoch}"] = current_state
        self.state_dict[self.current_epoch+epoch] = current_state

        if name:
            self.name = name

        if "legacy_name" in kwargs:
            # Hidden save method to save the pretrained models locally
            torch.save(self.state_dict, os.path.join(self.model_folder, f"{kwargs['legacy_name']}.pth"))
            self.model = model
        else:
            # Default behaving save method
            torch.save(self.state_dict, os.path.join(self.model_folder, f"ResNet{self.version}_{self.name}.pth"))
            with open(os.path.join(self.model_folder, f"ResNet{self.version}_{self.name}_LabelEncoder.json"), "w") as json_file:
                json.dump(self.label_encoder, json_file)

        return None
    

    def edit_optimizer(self, **kwargs):
        """
        Edits the model optimizer hyperparameters. Note that some optimizers take different parameters, 
        so the list below is not exhaustive. 

        This method won't save the model, so the optimizer parameter tuning will be acted only after a succesfull training or a 
        manual saving. In this last scenario, the previous optimizer parameters will be overwritten instead of saving a new version 
        at a later epoch. 

        Parameters
        ----------
        - lr: the learning rate
        - weight_decay: the weight_decay to apply to the targeted weights
        - ...
        """

        self.assert_model()
        if not self.current_epoch:
            print(f"ClassificationRESNET >> Failed to retreive a valid epoch to load the optimizer from")
            return False
        
        try:
            optimizer: torch.optim.Optimizer = self.state_dict[self.current_epoch]["optimizer"]
        except:
            print(f"VisionMODEL >> Failed to load the optimizer at epoch {self.current_epoch}")
            return False
        
        edited = False
        for param_group in optimizer.param_groups:
            for key, value in kwargs.items():
                if key in param_group:
                    param_group[key] = value
                    edited = True
                else:
                    print(f"VisionMODEL >> Parameter '{key}' is not a valid parameter for the optimizer")
        if edited:
            self.state_dict[self.current_epoch]["optimizer"] = optimizer
            print("VisionMODEL >> Optimizer parameters updated")
        return True


    def assert_model(self):
        """
        Asserts a deep learning model has been loaded 
        """
        if self.model is None:
            raise ValueError("No model was loaded, so this function can't be executed")
        return True
    

    def display_metrics(self, state_dict: dict = None, display: bool = False):
        """
        Plots the losses history of a model. Plots the checkpoint loss at which a model was saved.

        Parameters
        ----------
        - state_dict: the dictionnary of model 
        - display: whereas to show the plot in a pop-up window or not
        """

        if state_dict is None: state_dict = self.state_dict

        epochs = list(state_dict.keys())

        # print(epochs)
        # [print(len(state_dict[epoch]["loss_history"])) for epoch in epochs]

        yhistory = [state_dict[epoch]["loss_history"] for epoch in epochs[1:]]
        xhistory = [np.linspace(list(epochs)[epoch_idx], list(epochs)[epoch_idx+1], len(loss)) for loss, epoch_idx in zip(yhistory, range(len(epochs)-1))]

        xcheckpoints = epochs[1:] # 1 or 0 checkpoint per epoch
        ycheckpoints = [state_dict[epoch]["loss_checkpoint"] for epoch in epochs[1:]]
        # print(xcheckpoints, ycheckpoints)

        xhistory_plottable = []
        [[xhistory_plottable.append(tick) for tick in segment] for segment in xhistory]
        yhistory_plottable = []
        [[yhistory_plottable.append(loss) for loss in segment] for segment in yhistory]

        # print(len(xhistory_plottable), len(yhistory_plottable))
        # print(len(xcheckpoints), len(ycheckpoints))

        plt.figure(figsize=(16, 8))
        plt.semilogy(xhistory_plottable, yhistory_plottable, label="Loss history")
        plt.semilogy(xcheckpoints, ycheckpoints, "x-", color="red", label="Loss checkpoints")
        plt.legend()
        plt.xlabel("Epoch number")
        plt.ylabel("Loss value")
        plt.grid(True)
        plt.xticks(epochs, rotation=45)
        plt.title("Model loss history and saving checkpoints")
        plt.savefig(os.path.join(os.path.dirname(PRELABELLING_DIR_PATH), "outputs", "Training_report.png"))
        if display: plt.show()
        plt.close('all')

        return None


    def display_state_dict(self, state_dict: dict):
        """
        Simulates a __rep__ method for a dict of various items
        """

        _rep = ""
        for key, val in state_dict.items():
            print(key, type(key))
            _rep += "\n" + key + str(val)

        return _rep
    

    def classification_metrics(self, output, target, label_decoder = None):
        """
        Computes a score for a classification guess, depending on the model certainty of a label. 
        High values mean certain correct guesses, low mean certain incorrect guesses. Average is an uncertain
        guess, no matter the rightness of the guess.

        Returns
        -------
        accuracy: int
            O (wrong) or 1 (right)
        score: float
            score from 0 to 1 (higher is better)
        """
        if np.argmax(output) == np.argmax(target):
            return 1, max(output)
        else:
            return 0, 1 - max(output)


    def detection_metrics(self, pred_boxes, pred_labels, true_boxes, true_labels):
        """
        Computes the best iou of a box if the box is of a correct label
        Misses the negative impact of the boxes labelling a non existing target
        """
        iou = 0
        for true_box, true_label in zip(true_boxes, true_labels):
            # For every targets that should have been found
            ious = [0]
            for pred_box, pred_label in zip(pred_boxes, pred_labels):
                # For every point of interest detected by the model
                if pred_label == true_label:
                    # If the detected element is of interest, let's check the precision
                    iou1 = self._calculate_iou(box1=pred_box, box2=true_box)
                    iou2 = self._calculate_iou(box1=true_box, box2=pred_box)
                    # The supperposition is biaised by which box is compared to which
                    # So we only keep the smaller iou (i.e. a small box inside a large one shouldn't return 1.00 but the opposite) 
                    ious.append(min(iou1, iou2))
            # Let's only evaluate the accuracy of the best box
            iou += max(ious)
        
        return iou/len(true_boxes) # Returns the average iou


    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) of two bounding boxes.

        Args:
        - box1: First bounding box (numpy array of shape [4]).
        - box2: Second bounding box (numpy array of shape [4]).

        Returns:
        - iou: IoU score.
        """
        # Get coordinates of intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Compute area of intersection rectangle
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        
        # Compute areas of both bounding boxes
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        
        # Compute IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou


    def filter_top_predictions(self, boxes:np.ndarray, labels:np.ndarray, scores:np.ndarray, threshold:float=0.2, top_n:int=10):
        """
        Parses the SSD or FasterRCNN outputs (default is 200) to return only the ``top_n`` top scorers whose 
        confidence score is above the ``threshold``
        """
        mask = scores > threshold
        filtered_boxes = boxes[mask]
        filtered_labels = labels[mask]
        filtered_scores = scores[mask]
        
        sorted_indices = filtered_scores.argsort()
        
        top_boxes = filtered_boxes[sorted_indices][:top_n]
        top_labels = filtered_labels[sorted_indices][:top_n]
        top_scores = filtered_scores[sorted_indices][:top_n]
        
        return top_boxes, top_scores, top_labels


    def plot_image(self, output: list[float], filename: str, image: np.ndarray, label_encoder: dict = None):

        image = np.clip(image.transpose(1, 2, 0)*255 + 1 / 2, a_min=0, a_max=1)

        if label_encoder:
            label_decoder = {value: key for key, value in label_encoder.items()}
            title = f"Class: {label_decoder[np.argmax(output)]} | Score: {max(output):.4f}"

        plt.imshow(image)
        plt.title(title)
        plt.savefig(os.path.join(os.path.dirname(PRELABELLING_DIR_PATH), "outputs", f"{filename}.png")) 
        plt.close('all')  


    def plot_image_with_boxes(self, filename:str, image:np.ndarray, 
                              boxes:np.ndarray, labels:np.ndarray, scores:np.ndarray, 
                              true_boxes, true_labels,
                              label_encoder:dict=None, threshold:float=0.2):
        """
        Plot an image with bounding boxes and labels.
        
        Args:
            image (numpy.ndarray): The input image.
            boxes (numpy.ndarray): An array of shape (N, 4) containing the coordinates of N bounding boxes in format (xmin, ymin, xmax, ymax).
            labels (numpy.ndarray): An array of shape (N,) containing the class labels of the bounding boxes.
            scores (numpy.ndarray): An array of shape (N,) containing the confidence scores of the bounding boxes.
            label_encoder (dict): A dict where the keys are text classes and values the encoded label
            threshold (float): The confidence threshold for displaying bounding boxes.
        """

        image = np.clip(image.transpose(1, 2, 0)*255 + 1 / 2, a_min=0, a_max=1)

        class_names = None
        if label_encoder is not None:
            class_names = list(label_encoder.keys()) # reversed label encoder, to retreive text classes from int labels

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for box, label, score in zip(boxes, labels, scores):
            if score < threshold:
                continue
            
            xmin, ymin, xmax, ymax = box
            
            # Draw bounding box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            if class_names is not None:
                class_name = class_names[label-1] # 1 was added for background dummy label, now we remove it
            else:
                class_name = f'Class {label}'
            
            ax.text(xmin, ymin - 2, f'{class_name} {score:.2f}', color='r')
        
        for box, label in zip(true_boxes, true_labels):
            
            xmin, ymin, xmax, ymax = box
            
            # Draw bounding box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            if class_names is not None:
                class_name = class_names[label-1] # 1 was added for background dummy label, now we remove it
            else:
                class_name = f'Class {label}'
            
            ax.text(xmin, ymin - 2, f'{class_name}', color='g')

        plt.axis('off')
        plt.savefig(os.path.join(os.path.dirname(PRELABELLING_DIR_PATH), "outputs", f"{filename}.png")) 
        plt.close('all')


    def plot_image_with_mask(self, image:np.ndarray, mask:np.ndarray, filename:str):

        image = np.clip(image.transpose(1, 2, 0)*255 + 1 / 2, a_min=0, a_max=1)
        
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(mask, cmap='viridis', alpha=0.7)  # You can adjust alpha to make the mask more or less transparent

        plt.axis('off')
        plt.savefig(os.path.join(os.path.dirname(PRELABELLING_DIR_PATH), "outputs", f"{filename}.png")) 
        plt.close('all')
