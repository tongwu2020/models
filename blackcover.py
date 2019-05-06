
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


model = torch.load('/home/tong/glass/donemodel/model2.pkl')
# model = torch.load('/home/research/tongwu/glass/donemodel/model2.pkl')

data_dir = '/home/tong/glass/test'
image_datasets = datasets.ImageFolder(data_dir,
        transforms.Compose([
        transforms.Resize(size = (224,224)),
           transforms.ToTensor()]))


dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1,
                                             shuffle=True)
   

class_names = image_datasets.classes


def blackcover(model, X, y, width, height, xskip, yskip):
	#wideth:44 , height:22, xship:22. yship:22
    """ Construct FGSM adversarial examples on the examples X"""
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.ones_like(X).to(y.device)
    xtimes = 224//xskip
    ytimes = 224//yskip

    for i in range(xtimes):
        for j in range(ytimes):

            blackcover = np.ones([224,224,3]).astype(np.float32)*255
            blackcover[yskip*j:(yskip*j+height),xskip*i:(xskip*i+width),:] = 0 
            blackcover = transforms.ToTensor()(blackcover).to(y.device)

            #print(blackcover[:,1,1])
            # out = torchvision.utils.make_grid(blackcover)
            # imshow(out)
            

            all_loss = nn.CrossEntropyLoss(reduction='none')(model( X*blackcover), y )
            if(all_loss>=max_loss):
                max_delta = blackcover.detach()
                max_loss = torch.max(max_loss, all_loss)
        
    return max_delta



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu()
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(100)  # pause a bit so that plots are updated


def run(num_iter1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(class_names)
    print("success1")
    # glass1 = cv2.imread('/home/tong/glass/models/dataprepare/silhouette.png')
    # #np.set_printoptions(threshold=np.inf)
    # #print(glass)
    # glass = transforms.ToTensor()(glass1)
    # print(sum(sum(sum(glass))))

 

    count = 0
    total = 0
    black2= np.ones([224,224,3]).astype(np.float32)*255
    black3 = transforms.ToTensor()(black2).to(device)
    
    for data in dataloaders:
        images, labels = data
        #print(images)
        # glass = glass.to(device)        
        images = images.to(device)
        labels = labels.to(device)
        total += 1

        black = blackcover(model, images, labels,44,22,22,22)
        images = black*images
        outputs = model((images))
        _, predicted = torch.max(outputs.data, 1)
        count += (predicted == labels).sum().item()


        black3 = black3 - black*(1/100)
        if(total==101):
            break



       

            #print("suc")
            #inputs, classes = next(iter(dataloaders['train']))
            #Make a grid from batch
            #out = torchvision.utils.make_grid(images)
            #imshow(out, title=[class_names[x] for x in labels])
            #print(labels)
            #if labels.item()==0:
            #print("adv",predicted)       
        print(count/total,total)
    out = torchvision.utils.make_grid(black3)
    imshow(out)
        



    
if __name__ == "__main__":
    #for i in range(30):
        #print(i)
    for i in [1]:

        run(i)


