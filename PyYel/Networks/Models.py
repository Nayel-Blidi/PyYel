
import torch
import torch.nn as nn
import torch.nn.functional as F



def _flat_features_size(x) -> int:
    """
    Utility to compute the size of a feature of a batch.
    Takes a N-dimensions input, and computes the number of cells of the features.
    """
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return int(num_features)


class CNNx2(nn.Module):
    """
    A simple double-layers classifying CNN with a one hot encocoded output.
    \n
    Loss: CrossEntropyLoss
    Input dimensions: (batch, in_channels, height, width)
    \n
    Architecture:
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

    def __init__(self, in_channels=1, filters=16, hidden_layers=128, output_size=10, input_size=32, **kwargs):
        super(CNNx2, self).__init__()
        
        self.in_channels = in_channels
        self.outputsize = output_size
        self.filters = filters
        self.hidden_layers = hidden_layers

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=4*filters, kernel_size=3, padding=1, stride=1)

        self.linear1 = nn.Linear(_flat_features_size(torch.ones(in_channels, filters, input_size, input_size)), self.hidden_layers)
        self.linear2 = nn.Linear(self.hidden_layers, out_features=output_size)

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        
        x = x.view(-1, _flat_features_size(x))

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class CNNx3(nn.Module):
    """
    A simple triple-layers classifying CNN with a one hot encoded output.
    \n
    Loss: CrossEntropyLoss
    Input dimensions: (batch, in_channels, height, width)
    \n
    Architecture:
        Conv2d(filters, kernel=(3,3), padding=1, stride=1) -> RelU, Maxpool(2, 2), dropout(p), batchnorm ->\n
        Conv2d(4*filters, kernel=(3,3), padding=1, stride=1) -> RelU, Maxpool(2, 2), dropout(p), batchnorm -> \n
        Conv2d(4*filters, kernel=(3,3), padding=1, stride=1) -> ReLU, batchnorm, Maxpool(2, 2) -> Flatten -> \n
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

    def __init__(self, in_channels=1, filters=16, hidden_layers=128, output_size=10, input_size=32, **kwargs):
        super(CNNx3, self).__init__()
        
        self.in_channels = in_channels
        self.outputsize = output_size
        self.filters = filters
        self.hidden_layers = hidden_layers

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=filters)
        self.batchnorm2 = nn.BatchNorm2d(num_features=4*filters)
        self.batchnorm3 = nn.BatchNorm2d(num_features=8*filters)
        self.dropout = nn.Dropout(p=0.05)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=4*filters, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=4*filters, out_channels=8*filters, kernel_size=3, padding=1, stride=1)

        self.linear1 = nn.Linear(_flat_features_size(torch.ones(in_channels, filters, input_size, input_size)), self.hidden_layers)
        self.linear1 = nn.Linear(self.hidden_layers, self.hidden_layers)
        # self.linear2 = nn.Linear(self.hidden_layers, self.hidden_layers//4)
        self.linear3 = nn.Linear(self.hidden_layers, out_features=output_size)

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.batchnorm2(x)
        x = self.relu(self.conv3(x))
        x = self.batchnorm3(x)
        
        x = x.view(-1, _flat_features_size(x))

        x = self.linear1(x)
        x = self.relu(x)
        # x = self.linear2(x)
        # x = self.relu(x)
        x = self.linear3(x)

        return x
    

class ConnectedNNx3(nn.Module):
    """
    A simple 3 layers fully connected neural network.
    \n
    Regression:
        Input targets: 1D vector
        Loss: MSELoss    
    Binary classification: 
        Input targets: One-hot encoded classes
        Loss: BCELoss
    Multi classification:
        Input targets: One-hot encoded classes
        Loss: CrossEntropyLoss
     \n
    Architecture:
        Linear(input_size, hidden_layers) -> ReLU, batchnorm, dropout(p) ->\n
        Linear(hidden_layers, hidden_layers//4) -> ReLU, -> Linear(hidden_layers//4, output_size)
    \n
    Args:
        in_channels:
            Images: number for color channels. 1 for grayscale, 3 for RGB...
            Other: N/A
        filters: number of filters to apply and weighten (3x3 fixed kernel size)
        hidden_layers: classifying layers size/number of neurons
        output_size: number of labels, must be equal to the length of the one hot encoded target vector.
    """

    def __init__(self, input_size, output_size, hidden_layers=128, p=0., **kwargs):
        super(ConnectedNNx3, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

        self.input_layer = nn.Linear(input_size, hidden_layers)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_layers)
        
        self.layer2 = nn.Linear(hidden_layers, hidden_layers//4)
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_layers//4)

        self.output_layer = nn.Linear(hidden_layers//4, output_size)

    def forward(self, x):
        
        # Enforces the right input dimension (flattening)
        x = x.view(-1, self.input_size)

        # Features extraction
        # 1st layer
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        # 2nd layer
        x = self.layer2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)

        # Classifier
        # 3rd layer
        x = self.output_layer(x)  

        return x
    

class ConnectedNNx5(nn.Module):
    """
    A simple 5 layers fully connected neural network.
    \n
    Regression:
        Input targets: 1D vector
        Loss: MSELoss    
    Binary classification: 
        Input targets: One-hot encoded classes
        Loss: BCELoss
    Multi classification:
        Input targets: One-hot encoded classes
        Loss: CrossEntropyLoss
     \n
    Architecture:
        Linear(input_size, hidden_layers) -> ReLU, batchnorm, dropout(p) ->\n
        Linear(hidden_layers, hidden_layers//2) -> ReLU, batchnorm ->\n
        Linear(hidden_layers//2, hidden_layers//4) -> ReLU, batchnorm ->\n
        Linear(hidden_layers//4, hidden_layers//8) -> ReLU -> Linear(hidden_layers//8, output_size)
    \n
    Args:
        in_channels:
            Images: number for color channels. 1 for grayscale, 3 for RGB...
            Other: N/A
        filters: number of filters to apply and weighten (3x3 fixed kernel size)
        hidden_layers: classifying layers size/number of neurons
        output_size: number of labels, must be equal to the length of the one hot encoded target vector.
    """

    def __init__(self, input_size, output_size, hidden_layers=128, p=0., **kwargs):
        super(ConnectedNNx5, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

        self.input_layer = nn.Linear(input_size, hidden_layers)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_layers)
        
        self.layer2 = nn.Linear(hidden_layers, hidden_layers//2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_layers//2)
        self.layer3 = nn.Linear(hidden_layers//2, hidden_layers//4)
        self.batchnorm3 = nn.BatchNorm1d(num_features=hidden_layers//4)
        self.layer4 = nn.Linear(hidden_layers//4, hidden_layers//8)
        self.batchnorm4 = nn.BatchNorm1d(num_features=hidden_layers//8)

        self.output_layer = nn.Linear(hidden_layers//8, output_size)

    def forward(self, x):
        
        # Enforces the right input dimension (flattening)
        x = x.view(-1, self.input_size)

        # Features extraction
        # 1st layer
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        # 2nd layer
        x = self.layer2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        # 3rd layer
        x = self.layer3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)

        # Classifier
        # 4th layer
        x = self.layer4(x)
        x = self.relu(x)
        # 5th layer
        x = self.output_layer(x)  

        return x
    
