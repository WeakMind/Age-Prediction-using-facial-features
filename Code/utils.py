# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:47:43 2018

@author: h.oberoi
"""

import numpy as np
import pandas as pd
import os

def class_balance(path):
    file = pd.read_csv(path,delimiter=',')
    mat = file.values
    class_count = {}
    classes,counts = np.unique(mat[:,1],return_counts = True)
    for c,count in zip(classes,counts):
        class_count[str(c)] = int(count)
    print(class_count)
    
    

if __name__ == '__main__':
    class_balance(r'/media/Harshit/augmented.csv')