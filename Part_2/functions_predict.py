import torch
from torch import optim
import torchvision
from PIL import Image 
import numpy as np


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    #optimizer.load_state_dict = checkpoint['optimizer']
        
    return model


#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def process_image(image):
    
    with Image.open(image) as im:
        im = im.resize((256, 256))
        im = im.crop(((256-224)/2, (256-224)/2, 224 + (256-224)/2, 224 + (256-224)/2))
        im = np.array(im)/255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = (im - mean) / std
    
    return im.transpose(2,0,1)