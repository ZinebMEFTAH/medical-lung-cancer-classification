import os
import argparse

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from collections import Counter

from CnnGrayscaleModel import CnnGrayscaleModel
from MlpGrayscaleModel import MlpGrayscaleModel

# python3 sup_train.py --model MLP [--phase validation --epochs 10 --lr 0.0001 --batch_size 32]
# python3 sup_train.py --model CNN [--phase validation --epochs 10 --lr 0.0001 --batch_size 32]

def parse_arguments():
    """
    Parse command line arguments for model configuration.
    Returns:
        dict: Dictionary containing arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Choose the method to run MLP or CNN")
    parser.add_argument("--phase", type=str, required=False, help="Choose validation or test phase")
    parser.add_argument("--lr", type=float, required=False, help="Learning rate", default=1e-4)
    parser.add_argument("--epochs", type=int, required=False, help="Number of epochs", default= 10)
    parser.add_argument("--batch_size", type=int, required=False, help="Number data sample per batch", default= 32)
    return vars(parser.parse_args())


def set_seeds(seed):
    """
    Args:
        seed (int): Seed value for random number generators
    """
    torch.manual_seed(seed)


def prepare_dataset(img_size, dir_name):
    """
    Prepare and transform the CT scan dataset.
    Args:
        img_size (int): Target size for image resizing
        dir_name (string): Name of the directory containing the dataset split (train, val, test)
    Returns:
        dataset (ImageFolder) containing the prepared dataset and class names
    """
    dataset = datasets.ImageFolder(
        root="./ct_scan/"+dir_name,
        transform=v2.Compose([
            # TODO: Add necessary transforms
            
            v2.ToImage(),
            v2.Grayscale(num_output_channels=1),
            v2.Resize(size=(img_size, img_size)),
            v2.ToDtype(torch.float32, scale=True) # convert images to tensors, data structures used as entries by pytorch
        ])
    )
    
    return dataset


def create_data_loaders(batch_size):
    """
    Create train, validation and test data loaders.
    Args:
        batch_size (int): Number of samples per batch
    Returns:
        tuple: (train_loader, validation_loader, test_loader)
    """

    loaders = []
    for split in ['train', 'val', 'test']:
        dataset = prepare_dataset(IMG_SIZE, split)
        # TODO Print size of each dataset split and class names

        # --- Affichage des informations demandées ---
        print(f"\n--- Ensemble : {split} ---")
        # 1. Répartition par utilisation (taille totale du split)
        print(f"Nombre total d'images : {len(dataset)}") 
        
        # 2. Nom des classes
        print(f"Classes : {dataset.classes}") 
        
        # 3. Répartition des images par classe
        # dataset.targets contient l'indice de la classe pour chaque image
        counts = Counter(dataset.targets)
        for class_idx, count in counts.items():
            class_name = dataset.classes[class_idx]
            print(f" - Classe '{class_name}': {count} images") 

        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train')))
    
    return tuple(loaders)


def setup_model(setup_type, img_size, device):
    """
    Initialize either MLP or CNN model based on setup type.
    Args:
        setup_type (str): "MLP" or "CNN"
        img_size (int): Input image dimension
        device (str): Computing device ('cpu' or 'cuda')
    Returns:
        torch.nn.Module: Initialized model
    """
    model_class = MlpGrayscaleModel if setup_type == "MLP" else CnnGrayscaleModel
    model = model_class(nbr_classes=4, dimension=img_size, channels=1).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params} total parameters.")
    
    return model

def setup_training(model, learning_rate):
    """
    Configure training parameters and loss function.
    Args:
        model: Neural network model
        learning_rate (float): Learning rate for optimizer
    Returns:
        tuple: (optimizer, criterion)
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

#################
# Training pass
#################
def train(model, trainloader, optimizer, criterion):
    """
    Perform one epoch of training.
    Args:
        model: Neural network model
        trainloader: DataLoader for training data
        optimizer: Optimization algorithm
        criterion: Loss function
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    print("--- Training ---")
    
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # forward pass
        outputs = model(image)

        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # TODO: calculate the correct predictions
        data = outputs.data
        _, preds = torch.max(data, 1)
        train_running_correct += (preds == labels).sum().item()

        # backpropagation
        loss.backward()

        # update the optimizer parameters
        optimizer.step()

    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter

    # TODO: Calculate epoch accuracy
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))

    return epoch_loss, epoch_acc


#################
# Validation pass
#################
def validate(model, validation_loader, criterion):
    """
    Validate model performance on validation set.
    Args:
        model: Neural network model
        validation_loader: DataLoader for validation data
        criterion: Loss function 
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval() # Freeze the model
    print("--- Validation ---")
    
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad(): # Freeze the model

        for i, data in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            counter += 1
            
            image, labels = data
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            
            # TODO: calculate the correct predictions
            data = outputs.data
            _, preds = torch.max(data, 1)
            valid_running_correct += (preds == labels).sum().item()
            
    epoch_loss = valid_running_loss / counter

    # TODO: Calculate epoch accuracy
    epoch_acc = 100. * (valid_running_correct / len(validation_loader.dataset)) 

    return epoch_loss, epoch_acc

import matplotlib.pyplot as plt
import os

def plot_metrics(train_acc, valid_acc, train_loss, valid_loss, model):
    """
    Trace et sauvegarde les métriques d'entraînement et de validation.
    """
    os.makedirs("./plots", exist_ok=True)
    epochs_range = range(1, len(train_acc) + 1)

    # Création d'une figure avec deux graphiques côte à côte
    plt.figure(figsize=(12, 5))

    # --- Graphique 1 : Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Train Accuracy', color='blue', marker='o')
    plt.plot(epochs_range, valid_acc, label='Validation Accuracy', color='red', marker='x')
    plt.title(f'Accuracy over Epochs ({model})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # --- Graphique 2 : Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs_range, valid_loss, label='Validation Loss', color='red', marker='x')
    plt.title(f'Loss over Epochs ({model})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Ajustement automatique de l'espacement
    plt.tight_layout()

    # Sauvegarde du graphique
    save_path = f"./plots/metrics_{model}.png"
    plt.savefig(save_path)
    plt.close() # Ferme la figure pour libérer la mémoire
    print(f"Graphiques sauvegardés dans : {save_path}")



###########
# test pass
###########
def test(model, test_loader, model_name):
    """
    Test model performance on test set.
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        model_name: Model type ("MLP" or "CNN")
    """

    model.eval() # Freeze the model
    print("--- Test ---")
    
    test_running_correct = 0
    counter = 0

    with torch.no_grad(): # Freeze the model

        all_labels = []
        all_preds = []

        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            counter += 1
            
            image, labels = data
            all_labels.extend(labels.tolist())
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(image)

            # TODO: calculate the correct predictions
            data = outputs.data
            _, preds = torch.max(data, 1)
            test_running_correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())

    # accuracy for the complete epoch
    # TODO: Calculate epoch accuracy
    epoch_acc = 100. * (test_running_correct / len(test_loader.dataset)) 
    print(f"Test Accuracy for {model_name}: {epoch_acc:.3f} %")

    disp = ConfusionMatrixDisplay.from_predictions(
            all_labels, all_preds
        )
    disp.plot()
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/confusion_matrix_{model_name}.png")
    print('confusion matrix saved with the following labels: ')
    print('\t'+(', '.join(map(lambda kv: f"{kv[0]}: {kv[1]}", test_loader.dataset.class_to_idx.items()))))


if __name__ == "__main__":
    args = parse_arguments()

    DEVICE = 'cpu'
    IMG_SIZE = 512
    EPOCHS = args["epochs"]
    BATCH_SIZE = args["batch_size"]
    LEARNING_RATE = args["lr"]
    SEED = 42

    set_seeds(SEED)

    train_loader, validation_loader, test_loader = create_data_loaders(BATCH_SIZE)

    model = setup_model(args["model"], IMG_SIZE, DEVICE)
    optimizer, criterion = setup_training(model, LEARNING_RATE)

    if args["phase"] != "test":
        ###############################
        # Training and validation
        ###############################
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        for epoch in range(EPOCHS):

            print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
            
            train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            
            valid_epoch_loss, valid_epoch_acc = validate(model, validation_loader, criterion)
            valid_loss.append(valid_epoch_loss)
            valid_acc.append(valid_epoch_acc)

            print(f"[LOGS]: Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"[LOGS]: Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print("*"*50)
        
        plot_metrics(train_acc, valid_acc, train_loss, valid_loss, args["model"])
    
    else:
        ###############################
        # Training and test
        ###############################
        for epoch in range(EPOCHS):

            print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
            
            train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)

            print(f"[LOGS]: Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print("*"*50)

        test(model, test_loader, args["model"])
