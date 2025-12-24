import torch.nn as nn
import torch.nn.functional as F

# Exercice 5 : Configuration du modèle perceptron multi-couches (MLP)
class MlpGrayscaleModel(nn.Module):
    
    def __init__(self, nbr_classes, dimension, channels):

        super(MlpGrayscaleModel, self).__init__()

        self.dim = dimension
        self.channels = channels

        # TODO: Couche d'entrée prenant en entrée IMG_SIZE x IMG_SIZE x Number of channels x 128 (nombre de neurones de sortie de la couche d'entrée)
        self.fc1 = nn.Linear(dimension * dimension * channels, 128)
        # TODO: Couche intermédiaire prenant en entrée 128 neurones et en sortie 256 neurones
        self.fc2 = nn.Linear(128, 256)

        # TODO: Couche de classification (couche de sortie avec le nombre de classes que nous souhaitons classifier)
        self.fc_classification = nn.Linear(256, nbr_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        
        x = x.view(-1, self.dim * self.dim * self.channels)

        # Couche d'entrée prenant en entrée IMG_SIZE x IMG_SIZE x Number of channels x 128 (nombre de neurones de sortie de la couche d'entrée)
        x = self.fc1(x)
        # TODO: ReLU activation function
        x = F.relu(x)
        # TODO: Dropout layer
        x = self.dropout(x)

        
        # TODO: Couche intermédiaire prenant en entrée 128 neurones et en sortie 256 neurones
        x = self.fc2(x)
        # TODO: ReLU activation function
        x = F.relu(x)
        # TODO: Dropout layer
        x = self.dropout(x)
        
        # TODO: Couche de classification (couche de sortie avec le nombre de classes que nous souhaitons classifier)
        x = self.fc_classification(x)
        
        return x
