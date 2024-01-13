# PyYel
*PyYel* is a personnal library that aims at helping the deployement of strong data science tools, from data handling to deep learning.

To-do: (order of priority)
- Implement image augmentation
- Implement a support of pretrained neural networks
- Standardize the use of config files
- Deploy GUI (Tkinter) utilities to pilot the library
- Deploy HTML (dagster) utilities to pilot the pipelines
- Add TensorFlow support

Continuous development :
- Add new models of neural networks
- Improve the support of other data types

## Data
This module of the library is dedicated to data management. It possesses powerful utilities to manipulate the data, convert it, augment its size... It is also a good starting point to start implementing a data pipeline that will be fed to a neural network model.

#### Datapoint
A collection of classes that allow to convert a datapoint to a standardized format, that is expected by all the modules of the *PyYel* package.
- Datapoint : a class of usefool tools that simplifies the management and handling of a dataset. The inputs are converted to formats usable by the rest of the library. 
- Datatensor : a class of useful tools that simplifies the data preprocessing steps, in order to then deploy a machine learning solution using it. 

#### Utils
A collection of powerful tools that permit an easy manipulation of the datapoints.

#### Augmentations
A compilation of classes featuring methods to augment a datapoint of various type.
- ImageAugmentation : features a handfull of functions that can augment any type of data, as well as its labels. Can be deployed using the dedicated pipeline controller from the Utils.py file. 

## Networks
Networks regroups all the deep learning aspects of the PyYel package. It defines powerful models and pipelines that allow to easily deploy a machine learning tool to tackle a wide array of tasks and datasets.

#### Compiler
- Trainer : A training loop wrapper, that performs most of the steps required to train a model. 
- Tester : A testing loop wrapper, that performs most of the steps required to test a model.
- Loader : A model loading utility, that allows to easily retreive a saved model from its weight, and to deploy it somewhere else. 

#### Models
- CNNx2 : a simple two-layers convolutional network, that aims at solving multi-labels classification tasks 
- CNNx3 : a simple three-layers convolutional network, that aims at solving multi-labels classification tasks
- ConnectedNNx3 : A simple 3 layers fully connected neural network, that can handle a wide range of tasks.
- ConnectedNNx5 : A simple 5 layers fully connected neural network, that can handle a wide range of tasks.
