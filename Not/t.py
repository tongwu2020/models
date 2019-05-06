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



from art.attacks.carlini import CarliniL2Method




if __name__ == '__main__':
    model = torch.load('/home/research/tongwu/glass/donemodel/model2.pkl')

    data_dir = '/home/research/tongwu/glass/test'
    image_datasets = datasets.ImageFolder(data_dir,
            transforms.Compose([
            transforms.Resize(size = (224,224)),
            transforms.ToTensor()]))
    
    
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1,
                                                 shuffle=True)
       
    
    class_names = image_datasets.classes

    attack = CarliniL2Method()
    
    attack.generate(x_test[:100])


attacker = FastGradientMethod(classifier, eps=0.5)