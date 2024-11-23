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
ap = argparse.ArgumentParser(description='Predict.py')
ap.add_argument('input_img', help = 'Provide path to image. Mandatory argument', default='./flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', help = 'Provide path to checkpoint. This is a mandatory argument', default='./checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', help = 'Top K most likely classes. Optional', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', type=str)
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu_device", help='Use gpu or cpu for inference', type=str)

# a function that loads a checkpoint and rebuilds the model, input is a checkpoint path and output is a model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    if checkpoint['arch'] == 'allexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Architecture not recognized. Only 'alexnet', 'vgg16', and 'densenet121' are supported.")
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    for param in model.parameters():
        param.requires_grad = False
    return model

# function to process a PIL image for use in a PyTorch model, Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    width, height = pil_image.size
    if width > height:
        pil_image.thumbnail((50000, 256), Image.ANTIALIAS)
    else:
        pil_image.thumbnail((256, 50000), Image.ANTIALIAS)

    # Crop out the center 224x224 portion of the image
    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Normalize the image
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    return np_image.transpose(2, 0, 1)

# function to predict the class (or classes) of an image using a trained deep learning model, it takes an image path and a model and returns top k probabilities and the indices of those probabilities corresponding to the classes or class (it can be used for both)
def predict(image_path, model, topk=5, device='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    # Implement the code to predict the class from an image file
    if device == 'cuda':
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)

    image = image.unsqueeze(0)

    model.to(device)
    image.to(device)

    with torch.no_grad():
        output = model.forward(image)
    output_prob = torch.exp(output)
    probs, indices = output_prob.topk(topk)
    probs = probs.cpu().numpy().tolist()[0]
    indices = indices.cpu().numpy().tolist()[0]
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indices]
    classes = np.array(classes)
    return probs, classes
    

#setting values data loading
args = ap.parse_args()
filepath = args.input_img

#defining device: either cuda or cpu
if args.gpu_device == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
#loading JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#loading model from checkpoint provided
model = load_checkpoint(args.checkpoint)

#defining number of classes to be predicted. Default = 1
if args.top_k:
    n_classes = args.top_k
else:
    n_classes = 1
#calculating probabilities and classes
probs, classes = predict(filepath, model, n_classes, device)
#preparing class_names using mapping with cat_to_name
class_names = [cat_to_name[i] for i in classes]
# displaying results
for i in range(len(class_names)):
    print("Number: {}/{}.. ".format(i+1, n_classes),
          "Class name: {}.. ".format(class_names[i]),
          "Probability: {:.3f}..% ".format(probs[i]*100))
print("Done.")