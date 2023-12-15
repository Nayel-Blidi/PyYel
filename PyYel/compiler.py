
import numpy as np
import matplotlib.pyplot as plt

import NNmodels

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from tqdm import tqdm


class Trainer():

    def __init__(self, 
                 model, train_dataloader: torch.Tensor, test_dataloader: torch.Tensor,
                 model_name="model", num_epochs=10,
                 input_path="", output_path="",
                 **kwargs
                 ) -> None:
        
        self.model = model
        self.model_name = model_name
        self.num_epochs = num_epochs

        self.input_path = input_path
        self.output_path = output_path
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
    def runPipeline(self):
        trainer_pipeline = [
            self.train()
        ]
        for step in trainer_pipeline:
            step
        return None

    def train(self):
            
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train() 

        running_loss = 0.0
        losses_list = []
        for epoch in tqdm(range(self.num_epochs)):

            for inputs, targets in self.train_dataloader:

                optimizer.zero_grad()  
                outputs = self.model(inputs)  

                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                losses_list.append(loss.item())

        print(f"Final loss: {loss.item()}")

        plt.semilogy(losses_list)
        plt.title("Training losses")
        plt.xlabel("Epochs")
        plt.ylabel("Running loss")
        plt.savefig(f"{self.output_path}/Loss_{self.model_name}_{self.num_epochs}e.png")
        # plt.show()

        # torch.save(self.model.state_dict(), f"{self.output_path}/{self.model_name}_weights_{self.num_epochs}e.pth")
        torch.save(self.model, f"{self.output_path}/{self.model_name}_{self.num_epochs}e.pth")
        print(f"Finished training, {self.model_name}_{self.num_epochs}e.pth saved")

        return None


class Tester():

    def __init__(self, 
                 model, train_dataloader: torch.Tensor, test_dataloader: torch.Tensor,
                 **kwargs
                 ) -> None:
        
        self.model = model

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def runPipeline(self):
        tester_pipeline = [
            self.test()
        ]
        for step in tester_pipeline:
            step
        return None

    def test(self):

        self.model.eval()
        with torch.no_grad():
            for test_features, test_labels in self.test_dataloader: # Tests real accuracy

                test_labels = torch.argmax(test_labels, dim=1).numpy()

                test_predicted = torch.argmax(self.model(test_features), dim=1).numpy()

                test_success = np.sum(np.equal(test_predicted, test_labels))
                print(f"testing success rate: {test_success/test_predicted.shape[-1]:.4f}")

            for train_features, train_labels in self.train_dataloader: # Tests overfitting

                train_labels = torch.argmax(train_labels, dim=1).numpy()

                train_predicted = torch.argmax(self.model(train_features), dim=1).numpy()
                train_success = np.sum(np.equal(train_predicted, train_labels))
                print(f"training accuracy: {train_success/train_predicted.shape[-1]:4f}")

        # predict_values, pred_counts = np.unique(test_predicted, return_counts=True)
        # labels_values, test_counts = np.unique(test_labels, return_counts=True)
        # print(pred_counts)
        # print(test_counts)

class Loader():

    def __init__(self,
                 model_name, input_path,
                 in_channels=None, output_size=None, filters=None, hidden_layers=None, input_size=None,
                 **kwargs
                 ) -> None:
        
        self.model_name = model_name
        self.output_model_path = input_path


        self.in_channels = in_channels
        self.output_size = output_size
        self.filters = filters
        self.hidden_layers = hidden_layers
        self.input_size = input_size
    
    def getModel(self):
        self.runPipeline()
        return self.model
    
    def getWeights(self):
        self.runPipeline()
        return self.weights

    def runPipeline(self):
        load_model_pipeline = [
            self.loadModel(),
            self.loadWeights(),
        ]
        
        [step for step in load_model_pipeline]

        return None

    def loadModel(self):
            
        try: 
            self.model = torch.load(f"{self.output_model_path}/{self.model_name}")
            print(f"Loading {self.model_name} weights")
        except:
            None
        try: 
            self.model = torch.load(f"{self.output_model_path}/{self.model_name}.pth")
            print("Model loaded")
        except:
            None

        return self.model

    def loadWeights(self):
        
        if "CNN" in self.model_name:
            print(f"Loading {self.model_name} weights")
            # self.model = NNmodels.CNN(in_channels=self.in_channels, 
            #                           output_size=self.output_size, 
            #                           filters=self.filters,
            #                           hidden_layers=self.hidden_layers
            #                           )
            try: 
                self.weights = torch.load(f"{self.output_model_path}/{self.model_name}")
                self.model.load_state_dict(self.weights)
            except: None                
            try: 
                self.weights = torch.load(f"{self.output_model_path}/{self.model_name}.pth")
                self.model.load_state_dict(self.weights)
            except: None

        elif "LayeredNN" in self.model_name:
            print(f"Loading {self.model_name} weights")
            # self.model = NNmodels.LayeredNN(input_size=self.input_size,
            #                                 output_size=self.output_size,
            #                                 hidden_layers=self.hidden_layers
            #                                 )
            try: 
                self.weights = torch.load(f"{self.output_model_path}/{self.model_name}")
                self.model.load_state_dict(self.weights)
                print("Weights loaded")
            except: None                
            try: 
                self.weights = torch.load(f"{self.output_model_path}/{self.model_name}.pth")
                self.model.load_state_dict(self.weights)
                print("Weights loaded")
            except: None

        else:
            print("Model loading failed")

        return self.weights
