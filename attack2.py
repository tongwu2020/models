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

model_dir = '/home/research/tongwu/glass/donemodel/model2.pkl'
print(model_dir)

model = torch.load(model_dir)


def pgd(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X
    """
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        #print("reach")
        loss = nn.CrossEntropyLoss()(model(X + delta ), y)
        loss.backward()
        
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    #print(loss)
    return delta.detach()


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

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True)
              for x in ['train', 'val','test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

correct = 0
total = 0 


eps = [2/255,4/255,8/255,8/255,8/255,16/255]
alpha = [0.5/255,1/255,2/255,2/255,2/255,4/255]
itera = [20,20,7,20,20,20]
restart = [20,20,20,1,20,20]

for i in range(len(eps)):
    correct = 0 
    total = 0 
    for data in dataloaders['test']:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        nums = 0
        for j in range(restart[i]):
            delta = pgd(model, images, labels, eps[i], alpha[i] ,itera[i] , True)
            outputs = model(images + delta)
            _, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            nums += (predicted == labels).sum().item()
        #print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))
        if nums == restart[i]:
            correct +=1
        print(nums)
    print("eps is ",eps[i],", alpha is ",alpha[i],", iteration is ",itera[i])
    print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / 470))




'''
from PIL import Image
    adv_image = images + delta
    adv_image = adv_image.cpu()
    #adv_image1 = np.resize(adv_image.numpy(),(3,224,224)).transpose((1, 2, 0))
    #adv_image2 = np.clip(adv_image1,0,1)
    inp1 = torchvision.utils.make_grid(adv_image)
    inp = inp1.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    #print(np.shape(adv_image2))
    #adv_image3 = np.resize(adv_image2,(3,224,224))
    #print(np.shape(adv_image3))
    #adv_image4 = np.transpose()
    #scipy.misc.imsave('/home/research/tongwu/glass/adv_test/'+str(labels.item())+'_'+str(count)+'.jpg',adv_image2)
    plt.imsave('/home/research/tongwu/glass/adv_test/'+str(labels.item())+'_'+str(count)+'.jpg',inp)
    '''










