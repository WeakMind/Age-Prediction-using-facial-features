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
from torchvision import transforms
import training_accuracy
import validation_accuracy

from skimage import io



class Data(Dataset):
    def __init__(self,augment,transform):
        #self.train = train 
        self.augment = augment
        self.l = os.listdir(self.augment)[:-6000]
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
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.fc = nn.Linear(512,512)
        self.l = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512,3)    
        self.classifier = nn.Sequential(self.model,self.l,self.fc2)
        
    def forward(self,x):
        return self.classifier(x)
def one_hot_embedding(filenames,mapping):
    filenames = list(filenames)
    labels = []
    for file in filenames:
        labels.append(mapping[file])
    labels = torch.LongTensor(np.array(labels))
    return labels
    


def train():
    mean = (0.44377467,0.33807371,0.3065616)
    std = (0.1508224,0.13366607,0.130034323)
    
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
    
     
    l = [transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize(mean,std)]
    images = Data(read_dir_augment,transforms.Compose(l))
    loader = DataLoader(images,batch_size = 256,num_workers= 2,shuffle=True)
    
    
    total_epochs = 200
    learning_rate = 0.0001
    weight_decay = 0.0001
    #weights = [.001/class_count['MIDDLE'] , .001/class_count['OLD'], .001/class_count['YOUNG']]
    #class_weights = torch.FloatTensor(weights).cuda()
    
    # model = torchvision.models.alexnet(pretrained = True)
    # for param in model.parameters():
    #     param.required_grad = False
    # model.classifier[6] = nn.Linear(4096,3)
    
    model = torchvision.models.alexnet(pretrained = True)
    model.classifier[6] = nn.Linear(4096,3)
    model = model.cuda()
    
    '''trainable = []
    for params in model.parameters():
        if params.requires_grad == True:
            trainable.append(params)'''
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate , weight_decay = weight_decay)
    loss_function = nn.CrossEntropyLoss().cuda()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=25,gamma = 0.5)
    
    
    for epoch in range(total_epochs):
        scheduler.step()
        count = 0
        total_loss = 0
        for i,(filenames,images) in enumerate(loader):
            model.train()
            
            minibatch_X = images.cuda()
            forward = model(minibatch_X)
            
            minibatch_Y = one_hot_embedding(filenames,mapping).long().cuda()
            loss = loss_function(forward,minibatch_Y)
            total_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count+=1
        print('Epoch : {} , Loss : {}'.format(epoch,total_loss/count))    
        torch.save(model.state_dict(),os.path.join('./saved_weights_12','model_{}.ckpt'.format(epoch)))
        if epoch%2 == 0:
            training_accuracy.training_acc(epoch, model)
            validation_accuracy.validation_acc(epoch, model)
if __name__ == '__main__':
    train()