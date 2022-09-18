#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:29:44 2022

@author: akashchoudhuri
"""

from collections import Counter
import pandas as pd
def freq_is(data,s,items):
    freq_itemsets=[]
    
    ###############################################################
                            ##Base Case##
    ###############################################################
    c = Counter() # Generating Candidate tuples with support values
    for i in items:
        for d in data:
            if(i in d[1]):
                c[i]+=1
    p = Counter() # Pruned frequentist itemset
    for i in c:
        if(c[i] >= s):
            p[frozenset([i])]+=c[i]
    freq_itemsets.append(p)
    ###############################################################
                        ##Recursive Case##
    ###############################################################
    pl = p
    pos = 1
    count=2
    while (True):
        num_counts = set()
        temp = list(p)
        for i in range(0,len(temp)):
            for j in range(i+1,len(temp)):
                t = temp[i].union(temp[j])
                if(len(t) == count): # For each count value, we only generate itemsets of count's length
                    num_counts.add(temp[i].union(temp[j]))
        num_counts = list(num_counts)
        c = Counter() # Generating Candidate tuples with support values
        for i in num_counts:
            c[i] = 0
            for d in data:
                if(i.issubset(set(d[1]))):
                    c[i]+=1
        p = Counter() # Pruned frequentist itemset
        for i in c:
            if(c[i] >= s):
                p[i]+=c[i]
        freq_itemsets.append(p)
        if(len(p) == 0):
            break
        pl = p
        count+=1
    count_is={}
    for i in range(len(freq_itemsets)):
        count_is[i+1]=len(freq_itemsets[i])
    return count_is,freq_itemsets