# Package Imports: All the necessary packages and modules are imported in the first cell of the notebook
# Imports here

# #To add the plots in a browser interface if necessary
# %matplotlib inline                                   
# %config InlindeBackend.figure_format = 'retina'      #To ensure the plot looks the same across the web

import time                                          #The time function to time the runtime of a cell of a program structure. This function acts like a stopwatch
import torch                                         #The pytorch library
import glob                                          #To find files or folders that follow a specific pattern in this case the name of the images
import random                                        #random library to choose a random image while running the final cell
import json                                          #To work with json content i.e. the cat_to_name.json
import collections                                   #A single use while creating an ordered dictionary of pre-trained models in the training cell
 
import matplotlib.pyplot as plt                      #This library and method is used plot images and graphs in the notebooks
import torch.nn.functional as F                      #The functional method is called to apply certain functonalities such as log_softmax or dropout
import numpy as np                                   #numpy library to handel arrays
from train_input_args import train_input_args                            #File that will allow the users to provide an input while running this file

# from torchvision import datasets, models, transforms #The torchvision library is used to call various methods
#                                                      #datasets: To paralelly load prebuilt datasets such as MNIST, COCO, etc.
#                                                      #models: To import pretrained models such as vgg16, vgg16_bn, efficientnet19, densenet121m etc.
#                                                      #transforms: To transfomr as single image into various forms viz. flipping the image horizontally, resizing, rotating, cropping, etc. 
#                                                      #It also helps with sending the images into a tensor using ToTensor() and using Normalize we can set the shape of the image to certain values.
from torch import nn, optim                          #To import the nn method and optimizer(optim) method from torch library
                                                     #nn: This can be used to declare a model type and assigning suitable parameters to it.
                                                     #optim: To create an optimizer with certain funcitons such as Adam, SGD, Sparse, etc.
from PIL import Image                                #Pillow library to process images
from the_separator import the_separator
from main import validation_test, create_pretrained_model, checkpoint
from dataloader import dataloader

the_separator()
#Arguments
train_args = train_input_args()
print("Argument 1:", train_args.path)
print("Argument 2:", train_args.arch)
print("Argument 3:", train_args.learning_rate)
print("Argument 4:", train_args.hidden_units)
print("Argument 5:", train_args.training_epochs)
print("Argument 6:", train_args.gpu)

the_separator()

#Assigning a variable to each arg-input
path_of_parent = train_args.path
arch = train_args.arch
learning_rate = train_args.learning_rate
number_of_hidden_units = train_args.hidden_units
epochs = train_args.training_epochs
_gpu_ = train_args.gpu

train_loader, validation_loader, test_loader, train_data, validation_data, test_data = dataloader(path_of_parent)

# # Training data augmentation: torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping

# # Data normalization: The training, validation, and testing data is appropriately cropped and normalized

# # Data loading: The data for each set (train, validation, test) is loaded with torchvision's ImageFolder

# # Data batching: The data for each set is loaded with torchvision's DataLoader


# data_dir = '/kaggle/input/flowers'            #Name of the data directory. This directory contains images of flowers
# train_dir = data_dir + '/train' #Train directory in the data_dir. It contains the images which we will be using for training the newtork
# valid_dir = data_dir + '/valid' #Validation Directory in the data_dir. This contains the images which we will be using for the validation test after training the network.
# test_dir = data_dir + '/test'   #Test directory in the data_dir. This directory consists of the test dataset that we will be using for the final test of the netwoek.

# # TODO: Define your transforms for the training, validation, and testing sets

# #data_transforms contains the transformation parameters that we will use for transforming the images in the validation and test datasets.
# #the transformation of data_transformation will be different from the transformation applied on the training_data
# #A instructed the images are transformed into 3 sets and further normalized into above mentioned mean and standard deviations.
# data_transforms = transforms.Compose([
#     transforms.Resize(225),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
# ])

# #train_transforms contains the transformation of the images in the training dataset. The 3 sets of the transformations in the training dataset
# #are different from the validation and test transformations but the normalization part is smiliar
# #The reason behind two different sets for transformation is to test the learning accuracy of the netwok we are about to see.
# train_transforms = transforms.Compose([
#     transforms.RandomRotation(25),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
# ])

# # TODO: Load the datasets with ImageFolder
# #Creating data_sets and applying transformations to the images in the directories.
# #Storing the transformations in different data_sets for training, validation and test
# train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
# validation_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
# test_data = datasets.ImageFolder(test_dir, transform=data_transforms)


# # TODO: Using the image datasets and the trainforms, define the dataloaders
# #Loading the data in a loader variable using DataLoader.
# #For train_loader we use the train_data where we make a batch_size of 64 prior to shuffling the elemetns of the datasets.
# #For validation and test loaders we follow the same method as the train_loader except the shuffling part.
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

#If cuda or GPU is available use it else use the cpu computation power
if _gpu_ == True:
    device = torch.device("cuda")
    print("Running on:", device)
else:
    device = torch.device("cpu")
    print("Running on:", device)
the_separator()
#We use the try and except method to handel the error when we try to call the 
#create_pretrained_model function and pass an invalid model as its parameter
#If a type error emerges becuase that's most anticipated we give and exception along with an instruction
#in the form of print statement

try:
    model, optimizer = create_pretrained_model(number_of_hidden_units,learning_rate,arch)
    model.class_to_idx = train_data.class_to_idx
    print(model)    
    model.to(device)    
    criterion = nn.NLLLoss()
except TypeError:
    print("\nCheck your model name in create_pretrained_model()\n")
the_separator()


#Training the network on train_loader

#To record the time when the program started
#Number of times the network will be trained on the dataset
train_losses, valid_losses = [], []
for epoch in range(epochs):
    training_running_loss = 0
    batches = 0   
    for images, labels in train_loader:
        batches += 1
        
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        log_p = model(images)
        loss = criterion(log_p, labels)
        loss.backward()
        optimizer.step()
        
        training_running_loss += loss.item()
        
        #If a batch is completed in an epoch this condition will decide it and print it
        if batches % 10 == 0:            
            print(f"  Batch {epoch+1}.{batches}/{epochs}.. done")
            
    else:
        with torch.no_grad():
            model.eval()
            valid_loss, valid_accuracy = validation_test(model, validation_loader, criterion, device)
            valid_losses.append(valid_loss)
                
            model.train()
        #Saving the better model while training itself instead of saving it later
        previous_valid_loss = 0
        if not previous_valid_loss or valid_loss < previous_valid_loss:
            checkpoint(model, optimizer, "model_checkpoint.pth", epochs)
            previous_valid_loss = valid_loss
            
        train_losses.append(training_running_loss/batches)
            
        print(f"Epochs: {epoch+1}/{epochs}.",                 
            f"Trian Loss:{training_running_loss/batches:.3f}.",
            f"Validation Loss:{valid_loss/batches:.3f}",
            f"Validation Accuracy:{100*valid_accuracy:.2f}")
        the_separator()
the_separator()

#PLotting the 
# plt.plot(train_losses, label = "Traning Loss")
# plt.plot(valid_losses, label = "Validation Loss")
# plt.legend(frameon=False)

the_separator()

# TODO: Do validation on the test set
model.to(device)
model.eval()
with torch.no_grad():
    test_loss, test_accuracy = validation_test(model, test_loader, criterion, device)
    
print(f"Test Loss:{test_loss:.3f}.",
     f"Test Accuracy:{100*test_accuracy:.2f}.")
the_separator()

# TODO: Save the checkpoint 
checkpoint(model, optimizer, "model_checkpoint.pth", epochs)
the_separator()

print(f"The model has been trained on:{arch} architecture.\n With {number_of_hidden_units} number of hidden units and {epochs} epochs on a {device}.\n Now you can run the predict.py file.")


        
        
    
