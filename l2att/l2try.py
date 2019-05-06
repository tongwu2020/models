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
import shutil
import re
import numpy as np


from l2attclass import AttackCarliniWagnerL2



def run(attack):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.eval()
    print(class_names)
    print("success1")

    correct = 0
    total = 0
    
    for batch_idx, data in enumerate(dataloaders):
        print("batch_idx is ", batch_idx)
        images, labels = data
        #print(images)       
        images = images.to(device)
        labels = labels.to(device)
        count = 0 
        num = 1
        total += 1
        for i in range(num):

            input_adv, out, distance = attack.run(model, images, labels, batch_idx)
            #outputs = model((input_adv+1)*0.5)
            #outputs = model((images+torch.randn_like(images, device='cuda') * 0.25).clamp(0,1))
            #_, predicted = torch.max(outputs.data, 1)
            #count += (predicted == labels).sum().item()
            print("output", out, "distance",distance)


        if count == num:
            correct += 1
        print(count,correct/total,total)
        

    print("final accuacy", correct/470)


    

    

if __name__ == '__main__':
    model = torch.load('/home/research/tongwu/glass/donemodel/model2.pkl')

    data_dir = '/home/research/tongwu/glass/test'
    image_datasets = datasets.ImageFolder(data_dir,
            transforms.Compose([
            transforms.Resize(size = (224,224)),
            transforms.ToTensor()]))
    
    
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=16,
                                                 shuffle=True)
       
    
    class_names = image_datasets.classes

    attack = AttackCarliniWagnerL2(
        targeted= False,
        max_steps= 1000,
        search_steps= 6 ,
        cuda= True,
        debug= False)
    
    run(attack)


