#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:38:23 2022

@author: akashchoudhuri
"""

import pandas as pd
import numpy as np
def preprocess(path):
    df=pd.read_csv(path,sep='\t',header=None)
    data=[]
    for i in range(len(df)):
        part=[]
        part.append('D'+str(i))
        oth=df.loc[i].tolist()
        dat=[]
        for j in range(100):
            if oth[j]=='Up':
                dat.append('gene'+str(j+1)+'_'+'up')
            elif oth[j]=='Down':
                dat.append('gene'+str(j+1)+'_'+'down')
            else:
                print('Exception')
        dat.append(oth[100])
        part.append(dat)
        data.append(part)
    return data