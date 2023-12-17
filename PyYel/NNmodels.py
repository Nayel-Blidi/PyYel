
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class CNN(nn.Module):
    """
    A simple classifying CNN with a one hot encocoded output.
    \n
    Loss: CrossEntropyLoss
    Input dimensions: (batch, in_channels, height, width)
    \n
    Architechture:
        Conv2d(filters, kernel=(3,3), padding=1, stride=1) -> ReLU, Maxpool(2, 2) ->\n
        Conv2d(4*filters, kernel=(3,3), padding=1, stride=1) -> ReLU, Maxpool(2, 2) -> Flatten -> \n
        Linear(in_channels*filters*height*width, hidden_layers) -> ReLU -> Linear(hidden_layers, output_size) 
    \n

    Args:
        in_channels:
            Images: number for color channels. 1 for grayscale, 3 for RGB...
            Other: N/A
        filters: number of filters to apply and weighten (3x3 fixed kernel size)
        hidden_layers: classifying layers size/number of neurons
        output_size: number of labels, must be equal to the length of the one hot encoded target vector.
    """

    def __init__(self, in_channels=1, filters=16, hidden_layers=128, output_size=10, **kwargs):
        super(CNN, self).__init__()
        
        self.in_channels = in_channels
        self.outputsize = output_size
        self.filters = filters
        self.hidden_layers = hidden_layers

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=4*filters, kernel_size=3, padding=1, stride=1)

        self.linear1 = nn.Linear(self.num_flat_features(torch.ones(in_channels, filters, 32, 32)), self.hidden_layers)
        self.linear2 = nn.Linear(self.hidden_layers, out_features=output_size)

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        
        x = x.view(-1, self.num_flat_features(x))

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

    #Flattens along dim>=1
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNNmax(nn.Module):
    """
    A CNN with a maxpool output and no classifying layers.
    Loss: BCELoss
    """
    def __init__(self, input_size, in_channels=1, output_size=4, hidden_layers=64, filters=16, **kwargs):
        super(CNNmax, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.filters = filters

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=None, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=output_size)

        self.input = nn.Linear(in_features=input_size, out_features=hidden_layers)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=3)

        self.output = nn.Linear(in_features=6, out_features=output_size)
        

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.max(x, dim=2)[0]

        x = self.output(x)

        return x
    
class LayeredNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=128, **kwargs):
        super(LayeredNN, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.1)

        self.input_layer = nn.Linear(input_size, hidden_layers)
        self.batchnorm = nn.BatchNorm1d(num_features=hidden_layers)
        
        self.layer1 = nn.Linear(hidden_layers, hidden_layers//2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_layers//2)
        self.layer2 = nn.Linear(hidden_layers//2, hidden_layers//4)
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_layers//4)
        self.layer3 = nn.Linear(hidden_layers//4, hidden_layers//8)
        self.batchnorm3 = nn.BatchNorm1d(num_features=hidden_layers//8)

        self.output_layer = nn.Linear(hidden_layers//8, output_size)

    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)
        
        x = self.output_layer(x)  

        return x
