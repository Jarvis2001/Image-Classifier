import argparse

# Model architecture: The training script allows users to choose from at least two different architectures available from torchvision.models
# Model hyperparameters: The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs

def train_input_args():
    parse = argparse.ArgumentParser(description='Predict Images')
    parse.add_argument('--input_image', default='/home/workspace/ImageClassifier/image/test/11/image_03098.jpg', type=str, dest='input_image')
    parse.add_argument('--checkpoint', default='/home/workspace/ImageClassifier/', type=str, dest='checkpoint')
    parse.add_argument('--top_k', default='5', type=int, dest='top_k')
    parse.add_argument('--gpu', default=False, type=bool,dest='gpu')
    parse.add_argument('--test_path', default='/home/workspace/ImageClassifier/flowers', type=str,dest='test_path')

    return parse.parse_args()