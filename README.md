# Handwritten-Digits-Classifier
Design and train neural networks with PyTorch to classify handwritten digits using MNIST dataset.

## Project Overview

The goal of this project is to train a convolutional neural network (CNN) model to accurately classify handwritten digits from the MNIST dataset. The model is built using PyTorch, a popular deep learning framework, and trained using the Adam optimizer.

## Project Summary

### Step 1
* Load the dataset from torchvision.datasets.
* Use transforms or other PyTorch methods to convert the data to tensors, normalize, and flatten the data.
* Create a DataLoader for your dataset.

### Step 2
* Visualize the dataset using the provided function and either:
    * Your training data loader and inverting any normalization and flattening.
    * A second DataLoader without any normalization or flattening.
* Explore the size and shape of the data to get a sense of what your inputs look like naturally and after transformation. Provide a brief justification of any necessary preprocessing steps or why no preprocessing is needed.

### Step 3
* Using PyTorch, build a neural network to predict the class of each given input image.
* Create an optimizer to update your network’s weights.
* Use the training DataLoader to train your neural network.

### Step 4
* Evaluate your neural network’s accuracy on the test set.
* Tune your model hyperparameters and network architecture to improve your test set accuracy, achieving at least 90% accuracy on the test set.

### Step 5
* Use torch.save to save your trained model.


## Requirements
* Python (3.x)
* PyTorch (1.x)
* torchvision
* PIL
