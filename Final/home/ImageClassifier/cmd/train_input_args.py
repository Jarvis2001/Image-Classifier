import argparse

# Model architecture: The training script allows users to choose from at least two different architectures available from torchvision.models
# Model hyperparameters: The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs

def train_input_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', default='/home/workspace/ImageClassifier/flowers/', type=str, dest='path')
    parse.add_argument('--arch', default='densenet121', type=str, dest='arch')
    parse.add_argument('--learning_rate', default='0.005', type=float, dest='learning_rate')
    parse.add_argument('--hidden_unit', default='512', type=int, dest='hidden_units')
    parse.add_argument('--training_epochs', default='10', type=int, dest='training_epochs')
    parse.add_argument('--gpu', default=False, type=bool, dest='gpu')
    
    
    return parse.parse_args()