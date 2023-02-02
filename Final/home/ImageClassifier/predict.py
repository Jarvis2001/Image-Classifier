#####################################################################################################
#   Name       :Makarand More
#   Created On :13/01/2023
#   Last Edit  :02/02/2023
#   Description:The predict.py file is used to obtain the final 
#               output of a flower species after training the neural network. 
#               It loads the .pth file saved during training and allows the 
#               user to run the prediction on either a GPU or CPU. Additionally, 
#               the user has the option to save images of the final output as jpg 
#               files for visualization purposes.    
#####################################################################################################

# %matplotlib inline
# %config InlindeBackend.figure_format = 'retina'      #To ensure the plot looks the same across the web

import time  # The time function to time the runtime of a cell of a program structure. This function acts like a stopwatch
import torch  # The pytorch library
import glob  # To find files or folders that follow a specific pattern in this case the name of the images
import random  # random library to choose a random image while running the final cell
import json  # To work with json content i.e. the cat_to_name.json
# A single use while creating an ordered dictionary of pre-trained models in the training cell
import collections
import os  # For working with directories

# This library and method is used plot images and graphs in the notebooks
import matplotlib.pyplot as plt
# The functional method is called to apply certain functonalities such as log_softmax or dropout
import torch.nn.functional as F
import numpy as np  # numpy library to handel arrays


# The torchvision library is used to call various methods
from torchvision import datasets, models, transforms
# datasets: To paralelly load prebuilt datasets such as MNIST, COCO, etc.
# models: To import pretrained models such as vgg16, vgg16_bn, efficientnet19, densenet121m etc.
# transforms: To transfomr as single image into various forms viz. flipping the image horizontally, resizing, rotating, cropping, etc.
# It also helps with sending the images into a tensor using ToTensor() and using Normalize we can set the shape of the image to certain values.
# To import the nn method and optimizer(optim) method from torch library
from torch import nn, optim
# nn: This can be used to declare a model type and assigning suitable parameters to it.
# optim: To create an optimizer with certain funcitons such as Adam, SGD, Sparse, etc.
from PIL import Image  # Pillow library to handle images
from dataloader import dataloader
from predict_input_args import train_input_args
from the_separator import the_separator
from main import create_pretrained_model

the_separator()
# Arguments
predict_args = train_input_args()
print("Argument 1:", predict_args.input_image)
print("Argument 2:", predict_args.test_path)
print("Argument 3:", predict_args.checkpoint)
print("Argument 4:", predict_args.top_k)
print("Argument 5:", predict_args.gpu)
print("Argument 6:", predict_args.json_path)
the_separator()

input_image = predict_args.input_image
test_path = predict_args.test_path
checkpoint = predict_args.checkpoint
top_k = predict_args.top_k
_gpu_ = predict_args.gpu
json_path = predict_args.json_path

# train_loader, validation_loader, test_loader = dataloader(path_of_parent_dir=input_images)
data_dir = test_path
test_dir = data_dir + '/test'

# with open('E:/Code/AIandMLAWS/Project2Classifier/backups/home/ImageClassifier/cat_to_name.json', 'r') as f:
with open(json_path, 'r') as f:
    cat_to_name = json.load(f)

create_pretrained_model(512, 0.005)

# Loading checkpoints: There is a function that successfully loads a checkpoint and rebuilds the model.
# TODO: Write a function that loads a checkpoint and rebuilds the model


def load_checkpoint(filepath):
    model_checkpoint = torch.load(filepath)
    model, optimizer = create_pretrained_model(
        512, 0.005, model_checkpoint["base_model"])
    model.load_state_dict(model_checkpoint["model_state_dict"])
    optimizer.load_state_dict(model_checkpoint["optim_state_dict"])
    model.class_to_idx = model_checkpoint['class_to_idx']
    return model, optimizer

# model_check_point = "E:/Code/AIandMLAWS/Project2Classifier/backups/home/ImageClassifier/cmd/model_checkpoint.pth"
model, optimizer = load_checkpoint(checkpoint)
print(model)
print(optimizer)

# Image Processing: The process_image function successfully converts a PIL image into an object that can be used as input to a trained model.


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)

    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transformed_img = transform(img)
    return np.array(transformed_img)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# Checking if the above function works as intended


def multiple_images(test_dir, top_k):
    print(f"{top_k}(random) pre-processed flower images.")
    # print(test_dir)
    files = glob.glob(test_dir + "/*/*.jpg")    
    # print(files)
    images = [random.choice(files) for _ in range(top_k)]
    for image_path in images:
        processed_image = process_image(image_path)
        print("The files that system decided to predict:\n", image_path)
    return processed_image, images, files
    #     imshow(processed_image)


def single_image():
    print(f"The image file is located at{input_image}.")
    processed_image = process_image(input_image)
    print("The files that system decided to predict:\n", input_image)
    return processed_image

choice = int(input(
    f"If you want to predict the {input_image} image press 1.\nIf you want to predict random {top_k} images press {top_k}.\n"))
choice
if choice == 1:
    processed_image = single_image()
else:
    images, processed_image, files = multiple_images(test_dir,top_k)

the_separator()

if _gpu_ == True:
    device = torch.device("cuda")
    print("Running on:", device)
else:
    device = torch.device("cpu")
    print("Running on:", device)
the_separator()
# Class Prediction: The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image.

# The following function will be called if the user wishes to run the program on GPU


def predict_gpu(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''

    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    processed_image = process_image(image_path)
    tensor_image = torch.from_numpy(
        processed_image).type(torch.cuda.FloatTensor)
    tensor_image = tensor_image.unsqueeze_(0)

    tensor_image.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor_image)

        probs = torch.exp(outputs)
        top_p, top_class = probs.topk(topk, dim=1)

        # GPU
        # numpy doesn't support cuda so for getting top_k and top_class we will pass the array extraction through the cpu
        # This process is possible via pyCUDA if we had no choice but to use a GPU.
        top_p = top_p.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]

    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]
    return top_p, top_class

# The following function will be called if the user wishes to run the program on CPU
# def predict_cpu(image_path, model, topk=5):


def predict_cpu(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    processed_image = process_image(image_path)
    tensor_image = torch.from_numpy(processed_image).type(torch.FloatTensor)
    tensor_image = tensor_image.unsqueeze_(0)

    tensor_image.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(tensor_image)

        probs = torch.exp(outputs)
        top_p, top_class = probs.topk(topk, dim=1)

        if _gpu_ == False:
            top_p = top_p.numpy()[0]
            top_class = top_class.numpy()[0]
        else:
            top_p = top_p.cpu().numpy()[0]
            top_class = top_class.cpu().numpy()[0]
    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]
    return top_p, top_class


# Sanity Checking with matplotlib: A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names.
# TODO: Display an image along with the top 5 classes
print(
    f"The image files will be saved in the images directory in {os.getcwd()}.\nThe files contain the plots of the probabilities of the images with picture of the flower.\n")
print("The following list is the accuracy (in percentage).")
the_separator()


def plotting_saving_printing(images):
    if choice == 1:
        if _gpu_ == True:
            probs, classes = predict_gpu(input_image, model, top_k)
        else:
            probs, classes = predict_cpu(input_image, model, top_k)

        flower_names = [cat_to_name[str(c)] for c in classes]

        # Set up plot
        plt.figure(figsize=(6, 10))
        ax = plt.subplot(2, 1, 1)

        # Set up title        
        flower_num = input_image.split('/')[2]
        print(flower_num)
        title_ = flower_num

        # Plot flower
        img = process_image(input_image)
        imshow(img, ax, title=title_)

        # Plot probabilities
        ax = plt.subplot(2, 1, 2)
        ax.barh(np.arange(len(flower_names)), probs, align='center')
        ax.set_yticks(np.arange(len(flower_names)))
        ax.set_yticklabels(flower_names)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')

        plt.savefig(str(flower_names[0])+'.jpg', bbox_inches="tight")

        # result = print(f"{flower_names[0]}:\t", probs[0]*100, "%")
        i = 0
        while i < top_k:
            result = print("{} with a probability of {}%".format(flower_names[i], (probs[i]*100).round(2)))
            i += 1
        return result
    else:
        for image_path in images:
            #image_path = test_dir + "/80/image_01983.jpg"
            if _gpu_ == True:
                probs, classes = predict_gpu(image_path, model, top_k)
            else:
                probs, classes = predict_cpu(image_path, model, top_k)

            flower_names = [cat_to_name[str(c)] for c in classes]

            # Set up plot
            plt.figure(figsize=(6, 10))
            ax = plt.subplot(2, 1, 1)

            # Set up title
            # print(image_path)
            flower_num = image_path.split("\\")[-1::]
            print(flower_num)
            title_ = flower_num

            # Plot flower
            img = process_image(image_path)
            imshow(img, ax, title=title_)

            # Plot probabilities
            ax = plt.subplot(2, 1, 2)
            ax.barh(np.arange(len(flower_names)), probs, align='center')
            ax.set_yticks(np.arange(len(flower_names)))
            ax.set_yticklabels(flower_names)
            ax.invert_yaxis()
            ax.set_xlabel('Probability')
            
            
            image_dir = 'images/'
            plt.savefig(str(image_dir+flower_names[0])+'.jpg', bbox_inches="tight")           

            # result = print(f"{flower_names[0]}:\t", probs[0]*100, "%")
            i = 0
            while i < top_k:
                result = print("{} with a probability of {}%".format(flower_names[i], (probs[i]*100).round(2)))
                i += 1
            the_separator()
        return result

plotting_saving_printing(processed_image)
the_separator()
print("Prediction Successfully.")
