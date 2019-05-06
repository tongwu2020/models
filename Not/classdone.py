import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import numpy as np
import os
from IPython import embed
import numpy as np
import os.path
from vgg_face import VGG_16
import shutil
import re
import numpy as np







if __name__ == "__main__":
    model = VGG_16()
    model.load_weights()
    path1 = '/Users/wutong/Research/vgg-faces-utils/class/train/abigail_spencer'
    path2 = '/Users/wutong/Research/vgg-faces-utils/class/Not11'
    
    files= os.listdir(path1)
    cout = 0 
    for file in files: 

        print(file)
        img = os.path.join(path1,file)
        #im = cv2.imread('/Users/wutong/Research/vgg-faces-utils/output/images/'+file)
        im = cv2.imread(img)
        #print(im)
        #print(type(im))
        if im is None :
            old_file_path = img
            new_file_path = os.path.join(path2,file)
            shutil.move(old_file_path,new_file_path)
            print('fail1')
            continue
        if len(im) is 0:
            old_file_path = img
            new_file_path = os.path.join(path2,file)
            shutil.move(old_file_path,new_file_path)
            print('fail2')
            continue

    
        print("....................")
        #print(im.shape())
        im = cv2.resize(im,(224,224))
        print(np.shape(im))
        im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224)
        im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
        print(np.shape(im))

        preds = F.softmax(model(im), -1)
        values, indices = preds.max(-1)
        ind = indices.numpy()
        print(ind)


'''
        if ind != 11:
            old_file_path = img
            new_file_path = os.path.join(path2,file)
            shutil.move(old_file_path,new_file_path)

        cout = cout+1
        if cout ==1000:
            break#
            '''