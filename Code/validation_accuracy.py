# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 16:20:02 2018

@author: h.oberoi
"""

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from skimage import io



class Data(Dataset):
    def __init__(self,augment,transform):
        #self.train = train 
        self.augment = augment
        self.l = os.listdir(self.augment)[-6000:]
        #self.l = self.l + os.listdir(self.augment)[:-2000]
        self.len = len(self.l)
        self.transform = transform
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        image = io.imread(os.path.join(self.augment,self.l[index]))
        image = (self.transform((image)))
        image = image.numpy()
        #image = np.transpose(image,(1,2,0))
        return self.l[index],image
    
def one_hot_embedding(filenames,mapping):
    filenames = list(filenames)
    labels = []
    for file in filenames:
        labels.append(mapping[file])
    labels = torch.LongTensor(np.array(labels))
    return labels
    

def validation_acc(epoch,model):
    mean = (0.44339467,0.38607371,0.3435616)
    std = (0.16508224,0.14396607,0.13934323)
    
    #read_dir_train = r'/media/Harshit/training_data'
    read_dir_augment = r'/media/Harshit/augmented_data'
    
    #answer_train = r'/media/Harshit/train.csv'
    answer_augment = r'/media/Harshit/augmented.csv'
    
    #ans1 = pd.read_csv(answer_train,delimiter=',')
    #ans1 = ans1.values
    
    ans2 = pd.read_csv(answer_augment,delimiter = ',')
    ans2 = ans2.values
    
    answer = ans2
    class_labels = {'MIDDLE':0 , 'OLD':1, 'YOUNG':2}
    mapping = {}
    class_count = {}
    for row in answer:
        mapping[row[0]] = class_labels[row[1]]
        if row[1] in class_count.keys():
            class_count[row[1]] = class_count[row[1]] + 1
        else:
            class_count[row[1]] = 1
            
    l = [transforms.ToPILImage(),transforms.RandomCrop(224,pad_if_needed=True),transforms.ToTensor(),transforms.Normalize(mean,std)]
    images = Data(read_dir_augment,transforms.Compose(l))
    loader = DataLoader(images,batch_size = 256,num_workers= 2,shuffle=False)
    
    
    '''model = torchvision.models.alexnet(pretrained = False)
    for param in model.parameters():
        param.required_grad = False
    model.classifier[6] = nn.Linear(4096,3)
    model.load_state_dict(torch.load('./saved_weights_3/model_{}.ckpt'.format(epoch)))
    model = model.cuda()'''
    model.eval()
    
    
    correct = 0
    total = 0
    for i,(filenames,images) in enumerate(loader):
        minibatch_X = images.cuda()
        forward = model(minibatch_X)
        forward = F.softmax(forward,dim = 1)
        minibatch_Y = one_hot_embedding(filenames,mapping).long().numpy()
        predicted = (torch.argmax(forward,dim=1)).cpu().numpy()
        correct += (predicted == minibatch_Y).sum()
        total+=minibatch_Y.shape[0]
        #print(correct)
        #import pdb;pdb.set_trace()
    print('Validation Accuracy : {}'.format(100*correct/total))
        
    
    #print('Epoch : {} , Loss : {}'.format(epoch,total_loss/count))    
    #torch.save(model.state_dict(),os.path.join('./saved_weights','model_{}.ckpt'.format(epoch)))

if __name__ == '__main__':
    validation_acc()