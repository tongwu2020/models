
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
import numpy as np


model = torch.load('/home/research/tongwu/glass/donemodel/adv_model3.pkl')



data_dir = '/home/research/tongwu/glass/adv_test/'


adv_transforms = transforms.Compose([
        transforms.Resize(size = (224,224)), #(size=(224, 224)
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
image_datasets = datasets.ImageFolder(data_dir,transform=adv_transforms)
                               
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1,shuffle=True)

print("success1")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)





correct = 0
total = 0 

with torch.no_grad():
    for data in dataloaders:
        images, labels = data
        #print(images)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        #print("ss")
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(correct,total)
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
