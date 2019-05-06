import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import copy
import cv2
import torch.nn.functional as F
import torchfile
from vgg_face import VGG_16
import shutil
import re
import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse
from core import Smooth
from time import time
import datetime

model_dir = '/home/research/tongwu/glass/donemodel/adv_modelfinal11.pkl'
print(model_dir)

model = torch.load(model_dir)

def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


def pgd2(model, X, y, epsilon=0.5, alpha=0.01, num_iter=40, randomize=False, restarts = 20):
    """ Construct FGSM adversarial examples on the examples X
    """
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X)
    for i in range(restarts):

        if randomize:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 2 * epsilon - epsilon
        else:
            delta = torch.zeros_like(X, requires_grad=True)
        for t in range(num_iter):
            #print("reach")
            loss = nn.CrossEntropyLoss()(model(X + delta ), y)
            loss.backward()
            delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) 
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
    
            #delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta),y)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    return max_delta


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size = (224,224)), #(size=(224, 224)
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/home/research/tongwu/glass'

batch = 16

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch,
                                             shuffle=True)
              for x in ['train', 'val','test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

correct = 0
total = 0 


eps     = [0.5 , 1  , 1.5 , 2  , 2.5 , 3  ]
alpha   = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
itera   = [20   , 20 , 20  , 20 , 20  , 20 ]
restart = [1    , 1  , 1   , 1  , 1   , 1  ]

for i in range(len(eps)):
    correct = 0 
    total = 0 
    check = 0 
    for data in dataloaders['test']:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        check +=1

        delta = pgd2(model, images, labels, eps[i], alpha[i] ,itera[i] ,False, restart[i])
        outputs = model(images + delta)
        _, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total += (labels == labels).sum().item()
        
        #print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))
        if check % 50 == 0:
            print("acc ", correct/total, "cor ", correct, "total ", total)
            # print(check,"check")
    print("eps is ",eps[i],", alpha is ",alpha[i],", iteration is ",itera[i]," restart is ", restart[i])
    print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / 470))











