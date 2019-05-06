
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

#djfkldsajflkj


model = torch.load('/home/research/tongwu/glass/donemodel/adv_modelfinal13.pkl')
print("11")

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
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True)
              for x in ['train', 'val','test']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
class_names = image_datasets['train'].classes

print("success1")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




correct = 0 
total = 0
with torch.no_grad():
    for data in dataloaders['test']:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images+ torch.randn_like(images, device='cuda') * 0.25)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(correct/total)
print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))

#test

