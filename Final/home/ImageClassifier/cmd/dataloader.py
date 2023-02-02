import torch                                         #The pytorch library
from torchvision import datasets, transforms         #The torchvision library is used to call various methods
                                                     #datasets: To paralelly load prebuilt datasets such as MNIST, COCO, etc.
                                                     #models: To import pretrained models such as vgg16, vgg16_bn, efficientnet19, densenet121m etc.
                                                     #transforms: To transfomr as single image into various forms viz. flipping the image horizontally, resizing, rotating, cropping, etc. 
                                                     #It also helps with sending the images into a tensor using ToTensor() and using Normalize we can set the shape of the image to certain values.




# Training data augmentation: torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping

# Data normalization: The training, validation, and testing data is appropriately cropped and normalized

# Data loading: The data for each set (train, validation, test) is loaded with torchvision's ImageFolder

# Data batching: The data for each set is loaded with torchvision's DataLoader

def dataloader(path_of_parent_dir):
    data_dir = str(path_of_parent_dir)            #Name of the data directory. This directory contains images of flowers
    train_dir = data_dir + '/train'               #Train directory in the data_dir. It contains the images which we will be using for training the newtork
    valid_dir = data_dir + '/valid'               #Validation Directory in the data_dir. This contains the images which we will be using for the validation test after training the network.
    test_dir = data_dir + '/test'                 #Test directory in the data_dir. This directory consists of the test dataset that we will be using for the final test of the netwoek.

# TODO: Define your transforms for the training, validation, and testing sets

#data_transforms contains the transformation parameters that we will use for transforming the images in the validation and test datasets.
#the transformation of data_transformation will be different from the transformation applied on the training_data
#A instructed the images are transformed into 3 sets and further normalized into above mentioned mean and standard deviations.
    data_transforms = transforms.Compose([
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

#train_transforms contains the transformation of the images in the training dataset. The 3 sets of the transformations in the training dataset
#are different from the validation and test transformations but the normalization part is smiliar
#The reason behind two different sets for transformation is to test the learning accuracy of the netwok we are about to see.
    train_transforms = transforms.Compose([
        transforms.RandomRotation(25),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

# TODO: Load the datasets with ImageFolder
#Creating data_sets and applying transformations to the images in the directories.
#Storing the transformations in different data_sets for training, validation and test
    train_data = datasets.ImageFolder(train_dir, train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
#Loading the data in a loader variable using DataLoader.
#For train_loader we use the train_data where we make a batch_size of 64 prior to shuffling the elemetns of the datasets.
#For validation and test loaders we follow the same method as the train_loader except the shuffling part.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)
    return train_loader, validation_loader, test_loader, train_data, validation_data, test_data
