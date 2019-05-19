# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:20:40 2018

@author: h.oberoi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 16:20:02 2018

@author: h.oberoi
"""

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import Dataset
import os

from torchvision import transforms


from skimage import io



class Data(Dataset):
    def __init__(self,path,transform):
        self.path = path
        self.l = os.listdir(path)
        self.len = len(self.l)
        self.transform = transform
        
        
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        image = io.imread(os.path.join(self.path,self.l[index]))
        image = (self.transform((image)))
        image = image.numpy()
        image = np.transpose(image,(1,2,0))
        return self.l[index],image
        
    

def training_data():
    
    read_dir = r'/media/Harshit/Train'
    write_dir = r'./training_data'
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)
    
    l = [transforms.ToPILImage(),transforms.Resize(255),transforms.RandomCrop(255,pad_if_needed=True),transforms.ToTensor()]
    
    images = Data(read_dir,transforms.Compose(l))
    loader = torch.utils.data.DataLoader(images,batch_size = 1,num_workers= 4,shuffle=False)
    
    
    
    for i,(file_name,image) in enumerate(loader):
        
        #print(boolean[0],file_name[0],class_name[0])
        image = image.numpy()
        io.imsave(os.path.join(write_dir,str(file_name[0])),image[0])
        #import pdb;pdb.set_trace()
            
if __name__ == '__main__':
    training_data()