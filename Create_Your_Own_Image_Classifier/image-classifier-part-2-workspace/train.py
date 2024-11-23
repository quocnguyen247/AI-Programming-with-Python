import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser(description='Train.py')
parser.add_argument('data_dir', help = 'Provide data directory. Mandatory argument', nargs='*', action="store", default="./flowers/", type=str)
parser.add_argument('--save_dir', help = 'Vgg16 can be used if this argument specified, otherwise Alexnet will be used', help = 'Provide saving directory. Optional argument', action="store", default="./checkpoint.pth", type=str)
parser.add_argument('--arch', action="store", default="vgg", type=str)
parser.add_argument('--learning_rate', action="store", default=0.001, type=float)
parser.add_argument('--hidden_units', action="store", default=500, type=int)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--gpu', action="store", default="gpu", type=str)

#setting values data loading
argparse = parser.parse_args()
data_dir = argparse.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#defining device: either cuda or cpu
if argparse.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
#data loading
if data_dir:
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
# Mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# load model vgg, alexnet, per model in case hidden_units or not.
def load_model(arch, hidden_units):
    if arch == 'vgg16':
        model=models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.3)),
                          ('fc2', nn.Linear(4096, hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.3)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        else:
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.3)),
                          ('fc2', nn.Linear(4096, 2048)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.3)),
                          ('fc3', nn.Linear(2048, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    else:
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units:
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.3)),
                          ('fc2', nn.Linear(4096, hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.3)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        else:
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.3)),
                          ('fc2', nn.Linear(4096, 2048)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.3)),
                          ('fc3', nn.Linear(2048, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    return model

# Defining validation Function. will be used during training
def validation(model, validloader, criterion):
    model.to(device)
    val_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    return val_loss, accuracy

# Loading model using above defined function
model, arch = load_model(argparse.arch, argparse.hidden_units)
# Actual training of the model
# initializing criterion and optimizer
criterion = nn.NLLLoss()
if argparse.learning_rate:
    optimizer = optim.Adam(model.classifier.parameters(), lr=argparse.learning_rate)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.to(device)
# setting number of epochs to be run
if argparse.epochs:
    epochs = argparse.epochs
else:
    epochs = 7

print_every = 40
steps = 0

# runing through epochs
for e in range(epochs):
    running_loss = 0
    model.train()
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)*100))
            running_loss = 0
            model.train()
# saving trained Model
model.to('cpu')

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx

# second variable is a class name in numeric
# creating dictionary for model saving
checkpoint = {'classifier': model.classifier,'state_dict': model.state_dict(),'arch': arch,'mapping': model.class_to_idx}
# Saving trained model for future use
if argparse.save_dir:
    torch.save(checkpoint, argparse.save_dir+ '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')