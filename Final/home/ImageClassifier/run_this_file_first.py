#####################################################################################################
#   Name       :Makarand More
#   Created On :31/01/2023
#   Last Edit  :02/02/2023
#   Description:Instructions to execute this project.
#####################################################################################################

from os import getcwd, mkdir

try:
    mkdir('images')
    print("This is an Image Classifier.\nThis image classifier can predict any flower based on its image.\n"
    f"\nFirst we run the train.py file.\nThis file can accept certain arguments.\nFor Example:\n"
    f"\npython train.py --arch 'densenet121' --learning_rate '0.05' --hidden_unit '512' training_epochs '5'\n"
    f"\nIf you want help understand what the parameters are you can type:\n python predict.py --help.\n"
    f"\nHint: While training a low learning rate is better, a value of 0.005 or 0.003, don't go too low.\n"
    f"\nAfter you successfully train the network, you are ready to run the predict.py file.\n"
    f"\nSimilar to train.py predict.py can accept certain arguments.\n"
    f"\npython predict.py --input_image 'E:\Code\AIandMLAWS\Project2Classifier\backups\home\ImageClassifier\cmd\image_3010.jpg' --chekpoint 'E:\Code\AIandMLAWS\Project2Classifier\backups\home\ImageClassifier\cmd\model_checkpoint.pth' --top_k 5 --json_path 'E:\Code\AIandMLAWS\Project2Classifier\backups\home\ImageClassifier\cmd\cat_to_name.json' --gpu True\n"
    f"\nHint: Provide full path to a file with the file name."
    f"\nAfter running the file the image results will be stored in the image directory in the {getcwd()} directory.\n"
    f"Have Fun!")
except FileExistsError:
    print(f"Check if {getcwd()} has a folder called images. If the folder exists delete it.")

