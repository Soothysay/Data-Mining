#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:45:24 2022

@author: akashchoudhuri
"""
import pandas as pd
from utils.preprocess import preprocess
from src.parta import freq_is
def main(support,printing):
    path='data/associationruletestdata.txt'
    data=preprocess(path)
    items = []
    for i in data:
        for q in i[1]:
            if(q not in items):
                items.append(q)
    items=sorted(items)
    sup=support/100
    s = int(sup*len(data))
    counts,frequent_itemset=freq_is(data, s, items)
    if printing==1:
        print('Support is set to be %d percentage' %support)
        for i in counts:
            if counts[i]>0:
                print('Number of length-%d frequent itemsets:%d' %(i,counts[i]))
    return frequent_itemset,data,items,sup
support=70
_,_,_,_=main(support,1)