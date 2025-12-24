import torch.nn as nn
import torch.nn.functional as F

# Exercice 5 : Configuration du modèle convolutif (CNN)
class CnnGrayscaleModel(nn.Module):
    
    def __init__(self, nbr_classes, dimension, channels):

        super(CnnGrayscaleModel, self).__init__()

        self.dim = dimension
        self.channels = channels

        # TODO: 1st Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, stride=1, padding=1)

        # TODO: 2nd Convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # TODO: Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # TODO: Classification layer
        input_size = 16 * (dimension // 4) * (dimension // 4)
        
        self.fc_classification = nn.Linear(input_size, nbr_classes)        
        

    def forward(self, x):

        # TODO: 1st Convolutional layer
        x = self.conv1(x)
        # TODO: 1st ReLU activation function
        x = F.relu(x)
        # TODO: 1st Pooling layer
        x = self.pool(x)
        
        # TODO: 2nd Convolutional layer
        x = self.conv2(x)
        # TODO: 2nd ReLU activation function
        x = F.relu(x)
        # TODO: 2nd Pooling layer
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)

        # TODO: Classification layer (couche de sortie avec le nombre de classes que nous souhaitons classifier)
        x = self.fc_classification(x)
        
        return x
