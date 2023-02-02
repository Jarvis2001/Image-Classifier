#####################################################################################################
#   Name       :Makarand More
#   Created On :13/01/2023
#   Last Edit  :02/02/2023
#   Description:This code contains the main section of the program including the classifier, 
#               validation test, and checkpoint creator. After the dataloader is run and the 
#               data is received, the neural network is trained based on the data. During the 
#               training process, the best segment or epoch with the highest accuracy will be 
#               saved in the current working directory in .pth format. Upon completion of the 
#               training, the model's results will be validated using the validation_test function.
#####################################################################################################

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


from torchvision import datasets, models, transforms #The torchvision library is used to call various methods
                                                     #datasets: To paralelly load prebuilt datasets such as MNIST, COCO, etc.
                                                     #models: To import pretrained models such as vgg16, vgg16_bn, efficientnet19, densenet121m etc.
                                                     #transforms: To transfomr as single image into various forms viz. flipping the image horizontally, resizing, rotating, cropping, etc. 
                                                     #It also helps with sending the images into a tensor using ToTensor() and using Normalize we can set the shape of the image to certain values.
from torch import nn, optim                          #To import the nn method and optimizer(optim) method from torch library
                                                     #nn: This can be used to declare a model type and assigning suitable parameters to it.
                                                     #optim: To create an optimizer with certain funcitons such as Adam, SGD, Sparse, etc.
from PIL import Image                                #Pillow library to process images
from train_input_args import train_input_args
# TODO: Build and train your network
#This cell includes the model and classifier required for training the train_loader
#the create_pretrained_model function comes with parameter base_model vgg16 as a default parameter.

def create_pretrained_model(hidden_units ,learning_rate ,base_model = "vgg16"):
    #a dictionary of pre_trained models that we can use in the future to train the dataset
    #The models in the dictionary can be found at https://pytorch.org/vision/0.8/models.html
    
    supported_base_model_dict = {
        'vgg16':models.vgg16,
        'vgg16_bn':models.vgg16_bn,
        'vgg19':models.vgg19,
        'vgg19_bn':models.vgg19_bn,
        'densenet121':models.densenet121,
        'densenet161':models.densenet161,
        'densenet169':models.densenet169,
        'densenet201':models.densenet201
        #we can choose any model from the above choices to train the dataset
    }
    
    #every model requires an input feature
    #an input_feature_dictionary
    input_features_dict = {
        'vgg16':25088,
        'vgg16_bn':25088,
        'vgg19':25088,
        'vgg19_bn':25088,
        'densenet121':1024,
        'densenet161':1024,
        'densenet169':1664,
        'densenet201':1920
    }
    
    #Variable to store the name of the base_model input from the user
    base_model_function = supported_base_model_dict.get(base_model, None)
    
    #A condition to check if the base model given by the user is in the dictionary or not
    #If the base model is not present the following code will print the supported base models
    if not base_model_function:
        print(f"Invalid base_model: Try:\n{','.join(supported_base_model_dict.keys())}")
    
    model = base_model_function(pretrained=True)
    input_features = input_features_dict[base_model]
    
    #freezing the parameters of the feature extraction
    for param in model.parameters():
        param.requires_grad = False
    
    model.base_model = base_model
    
    #Creating a classfier using nn method of pytorch
    #this classifier contains ReLU, dropout, log_softmax and linear functions
#     classifier = nn.Sequential(collections.OrderedDict([
#         ('fc1',nn.Linear(input_features, 512)),
#         ('relu1',nn.ReLU()),        
#         ('dropout1',nn.Dropout(0.05)),                             #The value as a parameter is the probablility of the element being replaced with zero. The default value is 0.5
#         ('fc3',nn.Linear(512,102)),
#         ('output',nn.LogSoftmax(dim=1))
#     ]))
    class classifier(nn.Module):
        def __init__(self, input_features):
            super(classifier, self).__init__()
            self.fc1 = nn.Linear(input_features, hidden_units)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.05)            
            self.fc2 = nn.Linear(hidden_units,102)
            self.output = nn.LogSoftmax(dim=1)
        def forward(self,x):
            x = x.view(x.shape[0], -1)
            
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.output(x)
            
            return x
        
    model.classifier = classifier(input_features)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    return model, optimizer

#Creating a validaiton function for future use. The function will come with the parameres
#model(training_model), valid_loader(validation dataset), criterion(to minimize the loss), device(cpu or gpu)
def validation_test(model, valid_loader, criterion, device):
    valid_loss = 0
    valis_accuracy = 0
    for images, labels in valid_loader:
        #loading the images and lables in the available devices
        images, labels = images.to(device), labels.to(device)
        
        #calculating the probability
        log_p = model(images)
        loss = criterion(log_p, labels)
        
        valid_loss += loss.item()
        
        ps = torch.exp(log_p)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        valis_accuracy += equals.type(torch.FloatTensor).mean()
    return valid_loss/len(valid_loader), valis_accuracy/len(valid_loader) 


#Creating a funciton that saves a trained model
#The function contanins parameters such as model(the model used for training), optimizer, filepath(where to save the model)
#The checkpoint dictionary contains the architecture of the model with the state of the model, 
#the name of the base model, the class mapping label, the optimized state and the number of epochs

def checkpoint(model, optimizer, filepath, epochs):
    print("Saving Model... ")
    model_checkpoint = {
        'model_state_dict': model.state_dict(),
        'base_model': model.base_model,
        'class_to_idx':model.class_to_idx,
        'optim_state_dict':optimizer.state_dict(),
        'n_epochs':epochs
    }
    torch.save(model_checkpoint, filepath)

