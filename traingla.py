
from __future__ import print_function, division
import sys
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


plt.ion()   # interactive mode


sys.stdout.write('begin')

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

data_dir = '/home/research/tongwu/glass'   # change this
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True)
              for x in ['train', 'val','test']}




dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(class_names)
print(dataset_sizes)




# def pgd(model, X, y, epsilon, alpha, num_iter, randomize):
#     """ Construct FGSM adversarial examples on the examples """
#     if randomize:
#         delta = torch.rand_like(X, requires_grad=True)
#         delta.data = delta.data * 2 * epsilon - epsilon
#         delta.data = (delta.data +X).clamp(0,1)-X
#         #print(delta.data)
#     else:
#         delta = torch.zeros_like(X, requires_grad=True)
#     for t in range(num_iter):
#         #print("reach")
#         loss = nn.CrossEntropyLoss()(model(X + delta ), y)
#         loss.backward()

#         delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#         delta.data = (delta.data +X).clamp(0,1)-X
#         delta.grad.zero_()
#     #print(loss)
#     return delta.detach()



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
        delta.data = (delta + alpha*delta0).clamp(-epsilon,epsilon)
      
        #print("ss")
        #delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = ((X+delta).clamp(0,1)-X)*glass        
        
        #print(delta.data)
        #print(t)
        delta.grad.zero_()
    return delta.detach()





def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    glass1 = cv2.imread('/home/research/tongwu/glass/models/dataprepare/silhouette.png')
    glass = transforms.ToTensor()(glass1)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                #torch.save(model_ft, '/home/research/tongwu/glass/donemodel/adv_model_005'+str(epoch) +'.pkl')
                #torch.save(model_ft, '/home/research/tongwu/glass/donemodel/adv_model_006.pkl')
            running_loss = 0.0
            running_corrects = 0
            #print("begin")
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                glass = glass.to(device)
                #print(dataloaders["train"][0].size())
                # zero the parameter gradient
                #print(inputs.size())
                #print(labels.size())

                #print(2/255)
                delta = pgd_glass(model, inputs, labels, glass, epsilon=1, alpha=0.07843, num_iter=20,
                         randomize=True)               
                inputs = inputs + delta
                #inputs = torch.cat((inputs, (inputs+delta).clamp(0,1)), dim=0)
                #print(inputs)
                #labels = torch.cat((labels,labels),dim=0)
                #print(labels)
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #print(preds,labels)
                running_corrects += torch.sum(preds == labels.data)
                print(running_loss,running_corrects)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = torch.load('/home/research/tongwu/glass/donemodel/model2.pkl')
print("..............1...............")

print("..............2...............")

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0004, momentum=0.9,weight_decay=0.0002)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

#model_ft = nn.DataParallel(model_ft,device_ids=[0,1])

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=10)

print("..............3...............")

torch.save(model_ft, '/home/research/tongwu/glass/donemodel/adv_modelfinal15.pkl')

#visualize_model(model_ft)


