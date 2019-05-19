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
import pandas as pd
import torch

from torch.utils.data import Dataset,DataLoader
import os

from torchvision import transforms


from skimage import io



class Data(Dataset):
    def __init__(self,path,classes,mapping,transform):
        self.path = path
        self.l = os.listdir(path)
        self.len = len(self.l)
        self.transform = transform
        self.classes = classes
        self.mapping = mapping
        
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        image = io.imread(os.path.join(self.path,self.l[index]))
        name = self.l[index].split('.')
        image = (self.transform((image)))
        image = image.numpy()
        image = np.transpose(image,(1,2,0))
        
        if self.mapping[self.l[index]] == 'MIDDLE':
            return True,'MIDDLE',name[0]+'_'+str(np.random.randint(1,20))+'.'+name[1],image
        elif self.mapping[self.l[index]] == 'OLD':
            return True,'OLD',name[0]+'_'+str(np.random.randint(1,20))+'.'+name[1],image
        else:
            return True,'YOUNG',name[0]+'_'+str(np.random.randint(1,20))+'.'+name[1],image
            
        
    

def augmentation(classes,mapping):
    
    read_dir = r'/media/Harshit/Train'
    write_dir = r'augmented_data_2'
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir)
    jitter = transforms.ColorJitter(brightness=0.1*torch.randn(1),
         contrast=0.1*torch.randn(1),saturation=0.1*torch.randn(1),hue=0.1*torch.randn(1))
    l = [transforms.ToPILImage(),transforms.Resize((255,255)),
         jitter,
         transforms.RandomRotation(30),
         transforms.RandomCrop(224,pad_if_needed=True),
         transforms.ToTensor()]
    
    images = Data(read_dir,classes,mapping,transforms.Compose(l))
    loader = DataLoader(images,batch_size = 1,num_workers= 4,shuffle=True)
    
    new_mapping = []
    old_count = 20000
    middle_count = 20000
    young_count = 20000
    for j in range(100):
        for i,(boolean,class_name,file_name,image) in enumerate(loader):
            if boolean == True:
                #print(boolean[0],file_name[0],class_name[0])
                if class_name[0] == 'OLD':
                    if old_count > 0:
                        old_count-=1
                    else:
                        continue
                if class_name[0] == 'MIDDLE':
                    if middle_count > 0:
                        middle_count-=1
                    else:
                        continue
                if class_name[0] == 'YOUNG':
                    if young_count > 0:
                        young_count-=1
                    else:
                        continue
                
                new_mapping.append([file_name[0],class_name[0]])
                image = image.numpy()
                io.imsave(os.path.join(write_dir,str(file_name[0])),image[0])
                #import pdb;pdb.set_trace()
            #import pdb;pdb.set_trace()
        if old_count<=0 and middle_count<=0 and young_count<=0:
            break
    file = pd.DataFrame(new_mapping)
    file.to_csv('augmented_2.csv',index=False,header=False)
    
        
def read(filename):
    file = pd.read_csv(filename,delimiter = ',')
    file = file.values
    print(file.shape)
    classes = {}
    mapping = {}
    for row in file:
        mapping[row[0]] = row[1]
        if row[1] in classes.keys():
            classes[row[1]]=classes[row[1]] + 1
        else:
            classes[row[1]]=1
    return classes,mapping
    
            
if __name__ == '__main__':
    classes,mapping = read(r'/media/Harshit/train.csv')
    print(classes)
    augmentation(classes,mapping)