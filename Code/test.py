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
    def __init__(self,test,l,transform):
        self.test = test 
        self.l = list(l[:,0])
        self.len = len(self.l)
        self.transform = transform
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        image = io.imread(os.path.join(self.test,self.l[index]))
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
    
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.model = torchvision.models.alexnet(pretrained = True)
        self.features = self.model.features
        self.bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256**2,4096)
        self.fc2 = nn.Linear(4096,3)
        
    def forward(self,x):
        x = self.features(x)
        x = self.bn(x)
        x = x.view(x.size(0),256,6*6)
        x = torch.bmm(x,torch.transpose(x,1,2))/36
        x = x.view(x.size(0),256**2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def test():
    mean = (0.44377467,0.33807371,0.3065616)
    std = (0.1508224,0.13366607,0.130034323)
    
    read_dir = r'/media/Harshit/Test'
    test_file = r'/media/Harshit/test.csv'
    class_labels = {0 : 'MIDDLE' , 1:'OLD', 2:'YOUNG'}
    testing = pd.read_csv(test_file,delimiter=',')
    testing = testing.values
    
    answer = []
    p = transforms.ColorJitter(brightness=0.1*torch.randn(1),contrast=0.1*torch.randn(1),saturation=0.1*torch.randn(1),hue=0.1*torch.randn(1))
    l = [transforms.ToPILImage(),transforms.Resize((224,224)),p,transforms.ToTensor(),transforms.Normalize(mean,std)]
    images = Data(read_dir,testing,transforms.Compose(l))
    loader = DataLoader(images,batch_size = 1,num_workers= 4,shuffle=False)
    
    
    #model = AlexNet()
    model = AlexNet()
    model.load_state_dict(torch.load('./saved_weights_13/model_50.ckpt'))
    model = model.cuda()
    model.eval()
    
    
    for i,(filenames,images) in enumerate(loader):
        minibatch_X = images.cuda()
        forward = model(minibatch_X)
        forward = F.softmax(forward,dim = 1)
        var = int(torch.argmax(forward,dim=1))
        s = filenames[0]
        
        #print(s)
        answer.append([class_labels[var],s])
    file = pd.DataFrame(answer)
    file.to_csv('test.csv',index=False,header=('Class','ID'))
    #print('Epoch : {} , Loss : {}'.format(epoch,total_loss/count))    
    #torch.save(model.state_dict(),os.path.join('./saved_weights','model_{}.ckpt'.format(epoch)))

if __name__ == '__main__':
    test()