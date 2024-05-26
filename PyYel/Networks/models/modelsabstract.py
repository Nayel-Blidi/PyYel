from abc import ABC, abstractmethod
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from sklearn.metrics import average_precision_score, precision_recall_curve
import shutil

NETWORKS_DIR_PATH = os.path.dirname(os.path.dirname(__file__))

class ModelsAbstract(ABC):
    """
    An AbstrcatBase Class that defines mandatory methods that should be found
    in every models classes.
    """
    def __init__(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass

    @abstractmethod
    def evaluate_datapoint(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    def __dir__(self):
        return ['load_model', 'sample_batch', 'train_model', 'test_model', 'save_model', 'evaluate_datapoint']
    
    def assert_model(self, model):
        """
        Asserts a deep learning model has been loaded 
        """
        if model is None:
            raise ValueError("No model was loaded, so this function can't be executed")
        return True
    
    def purge_weights(self, force=False):
        """
        Clears the .pth files of all the ``/weights`` and ``/weights_legacy`` folders.
        If ``force=True``, clears all the files indinsctively.
        """
        
        folders_paths = [
            os.path.join(NETWORKS_DIR_PATH, "models", "vision", "weights"),
            os.path.join(NETWORKS_DIR_PATH, "models", "vision", "weights_legacy"),
            os.path.join(NETWORKS_DIR_PATH, "models", "nlp", "weights"),
            os.path.join(NETWORKS_DIR_PATH, "models", "nlp", "weights_legacy"),
            os.path.join(NETWORKS_DIR_PATH, "models", "signal", "weights"),
            os.path.join(NETWORKS_DIR_PATH, "models", "signal", "weights_legacy"),
        ]

        for folder_path in folders_paths:

            # Check if the folder exists
            if not os.path.exists(folder_path):
                print(f"The folder {folder_path} does not exist.")
                return None

            # Iterate over all the files and directories in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    if file_path.endswith(".pth") or force:
                        os.unlink(file_path)  # Remove file or link
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

        return None

    def classification_metrics(self, pred, target):
        pass

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

    def plot_image(self, filename:str, output, image:np.ndarray, labels:list, label_encoder:dict=None):

        image = np.clip(image.transpose(1, 2, 0)*255 + 1 / 2, a_min=0, a_max=1)

        if label_encoder:
            class_names = list(label_encoder.keys()) # reversed label encoder, to retreive text classes from int labels
        
        title = ""
        for position, label in enumerate(labels):
            title += f"{position+1}: {output[position]:.2f} {class_names[label]}, "

        plt.imshow(image)
        plt.title(title)
        plt.savefig(os.path.join(f"{filename}")) 
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
        plt.savefig(filename) 
        plt.close('all')

        return fig

