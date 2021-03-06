import argparse
from core import Smooth
from time import time
import torch
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy

import cv2
import torch.nn.functional as F
import torchfile
import numpy as np
import shutil
import re



parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=32, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()
def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def pgd2(model, X, y, epsilon=0.5, alpha=0.01, num_iter=40, randomize=False, restarts = 20):
    """ Construct FGSM adversarial examples on the examples X
    """
    # max_loss = torch.zeros(y.shape[0]).to(y.device)
    # max_delta = torch.zeros_like(X)
    # for i in range(restarts):

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
        # all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta),y)
        # max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        # max_loss = torch.max(max_loss, all_loss)

    return delta.detach()


if __name__ == "__main__":
    # load the base classifier

    base_classifier = torch.load('/home/research/tongwu/glass/donemodel/model2.pkl')
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
    
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                  shuffle=True)
                   for x in ['train', 'val','test']}
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    
    print("success1")
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(len(image_datasets["test"]))
    smoothed_classifier = Smooth(base_classifier, 10, args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    
    dataset = image_datasets["test"]


    eps     = [0.5 , 1  , 1.5 , 2  , 2.5 , 3  ]
    alpha   = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    itera   = [20   , 20 , 20  , 20 , 20  , 20 ]
    restart = [1    , 1  , 1   , 1  , 1   , 1  ]

    #i =5

    for i in range(len(eps)):
    #if i ==5:
        cor = 0
        tot = 0 
        for k in dataloaders['test']:

        # only certify every args.skip examples, and stop after args.max examples
        # if k % args.skip != 0:
        #     continue
        # if k == args.max:
        #     break

            (x, label) = k
            x = x.to(device)
            labels = label.to(device)
            # print(label)
    
            before_time = time()
            
            delta = pgd2(base_classifier, x, labels, eps[i], alpha[i] ,itera[i] ,False, restart[i])
            x = x + delta
            
            # make the prediction
            prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)
    
            after_time = time()
            correct = int(prediction == int(label))

            # print(alpha[i],args.alpha)
            #print(prediction,label)

    
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    
            # log the prediction and whether it was correct
            print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction, correct, time_elapsed), file=f, flush=True)
            if correct==1 :
                cor +=1
            tot+=1
    
        print(cor/tot)

    f.close()


