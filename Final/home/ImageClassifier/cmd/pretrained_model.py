def create_pretrained_model(base_model = "vgg16"):
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
            self.fc1 = nn.Linear(input_features, 512)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.05)            
            self.fc2 = nn.Linear(512,102)
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
    
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
    
    return model, optimizer