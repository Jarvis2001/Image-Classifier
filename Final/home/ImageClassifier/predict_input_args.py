#####################################################################################################
#   Name       :Makarand More
#   Created On :13/01/2023
#   Last Edit  :02/02/2023
#   Description:This code in predict_input_args.py allows the user 
#               to specify certain arguments when running the 
#               predict.py file. These arguments include the usage 
#               of a GPU (cuda) device or a CPU, as well as the 
#               location of the checkpoint file to be used. The 
#               user can specify these arguments by using the 
#               --gpu option, followed by either True or False, 
#               depending on their desired device.
#####################################################################################################

import argparse
import pathlib

# Model architecture: The training script allows users to choose from at least two different architectures available from torchvision.models
# Model hyperparameters: The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
print(pathlib.Path('image_08106.jpg'))
def train_input_args():
    parse = argparse.ArgumentParser(description='Predict Images')
    parse.add_argument('--input_image', default='./inputimage/image_03130.jpg', type=str, dest='input_image', help='Provide an image in jpg format saved in current working directory.')
    parse.add_argument('--checkpoint', default='./model_checkpoint.pth', type=str, dest='checkpoint')
    parse.add_argument('--top_k', default='5', type=int, dest='top_k')
    parse.add_argument('--gpu', default=False, type=bool,dest='gpu')
    parse.add_argument('--test_path', default='E:\Code\AIandMLAWS\Project2Classifier\backups\home\ImageClassifier\cmd', type=str,dest='test_path')
    parse.add_argument('--json_path', default='./cat_to_name.json', type=str, dest='json_path',help="Enter the location of the .json file")

    return parse.parse_args()