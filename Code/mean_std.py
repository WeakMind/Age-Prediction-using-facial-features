# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 16:20:02 2018

@author: h.oberoi
"""

import numpy as np
from torch.utils.data import Dataset,DataLoader
import os

from torchvision import transforms


from skimage import io



class Data(Dataset):
    def __init__(self,test,transform):
        self.test = test 
        #self.augment = augment
        self.l = os.listdir(self.test)
        #self.l = self.l + os.listdir(self.augment)
        self.len = len(self.l)
        self.transform = transform
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        image = io.imread(os.path.join(self.test,self.l[index]))
        image = (self.transform((image)))
        image = image.numpy()
        #image = np.transpose(image,(1,2,0))
        return image
    

if __name__ == '__main__':
    mean = (0.43987089,0.33920664,0.30778521)
    std = (0.15444025,0.13871018,0.13485506)
    
    read_dir_train = r'/media/Harshit/augmented_data'
    #read_dir_augment = r'/media/Harshit/augmented_data'
    
    l = [transforms.ToTensor()]
    images = Data(read_dir_train,transforms.Compose(l))
    loader = DataLoader(images,batch_size = 1,num_workers= 4,shuffle=False)
    
    running_mean = np.zeros(shape = (1,3))
    running_std = np.zeros(shape = (1,3))
    count=0
    for i,image in enumerate(loader):
        image = image.numpy()
        running_mean = running_mean + (np.mean(image,axis = (2,3)))
        running_std = running_std + (np.std(image,axis = (2,3)))
        count+=1
        
    print(running_mean/count)
    print(running_std/count)