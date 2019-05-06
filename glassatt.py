from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import torch.nn.functional as F
import torchfile
import numpy as np
from vgg_face import VGG_16
import shutil
import re


model = torch.load('/home/tong/glass/donemodel/adv_modelfinal15.pkl')
# model = torch.load('/home/research/tongwu/glass/donemodel/model2.pkl')

data_dir = '/home/tong/glass/test'
image_datasets = datasets.ImageFolder(data_dir,
        transforms.Compose([
        transforms.Resize(size = (224,224)),
           transforms.ToTensor()]))


dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1,
                                             shuffle=True)
   

class_names = image_datasets.classes


def pgd_glass(model, X, y, glass,epsilon=0.1, alpha=0.01, num_iter=20, randomize=True):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
        delta.data = (delta.data+X).clamp(0,1)-X
    else:
        delta = torch.zeros_like(X, requires_grad=True)
    
    for t in range(num_iter):
        
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        #print(torch.max(torch.max(delta.grad.detach())))
        delta0 = delta.grad.detach()/torch.max(torch.max(delta.grad.detach()))
        # print(delta0)
        delta.data = (delta + alpha*delta0).clamp(-epsilon,epsilon)
      
        #print("ss")
        #delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = ((X+delta).clamp(0,1)-X)*glass        
        
        #print(delta.data)
        #print(t)
        delta.grad.zero_()
    return delta.detach()



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu()
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    # inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


def run(num_iter1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(class_names)
    print("success1")
    glass1 = cv2.imread('/home/tong/glass/models/dataprepare/silhouette.png')
    #np.set_printoptions(threshold=np.inf)
    #print(glass)
    glass = transforms.ToTensor()(glass1)
    print(sum(sum(sum(glass))))
    print(np.shape(glass1))

 

    correct = 0
    total = 0
    
    for data in dataloaders:
        images, labels = data
        #print(images)
        glass = glass.to(device)        
        images = images.to(device)
        labels = labels.to(device)
        count = 0 
        num = 5
        total += 1
        for i in range(num):
            images = images + pgd_glass(model, images, labels, glass, epsilon=1,
             alpha=0.07843, num_iter=num_iter1, randomize=True)
            outputs = model((images).clamp(0,1))
            _, predicted = torch.max(outputs.data, 1)
            count += (predicted == labels).sum().item()
       
            # out = torchvision.utils.make_grid(images)
            # imshow(out, title = [class_names[x] for x in labels])
            #print("suc")
            #inputs, classes = next(iter(dataloaders['train']))
            #Make a grid from batch
            #out = torchvision.utils.make_grid(images)
            #imshow(out, title=[class_names[x] for x in labels])
            #print(labels)
            #if labels.item()==0:
            #print("adv",predicted)       
        

        if count == num:
            correct += 1
        print(count,correct/total,total)
        

    print("final accuacy", correct/470)


    
if __name__ == "__main__":
    #for i in range(30):
        #print(i)
    for i in [1,2,3,5,7,10,20]:

        run(i)


