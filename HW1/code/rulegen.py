#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:45:45 2022

@author: akashchoudhuri
"""

import pandas as pd
from itertools import combinations
from main_freq_itemset import main
def rg(support,confidence):
    
    frequent_itemset,data,items,sup=main(support,0)
    rules=set()
    for p in range(1,(len(frequent_itemset))):
        prev_l=frequent_itemset[p]
        
        for l in prev_l:
            combi=[frozenset(q) for q in combinations(l, len(l)-1)] # Generating all combinations of length l-1
            for ele in combi:
                ab=0
                ba=0
                excl=l-ele
                whole=l
                count_whole=0
                count_excl=0
                count_ele=0
                for d in data:
                    if ele.issubset(set(d[1])):
                        count_ele+=1
                    if whole.issubset(set(d[1])):
                        count_whole+=1
                    if excl.issubset(set(d[1])):
                        count_excl+=1
                        
                conf1=(count_whole/count_ele)*100
                conf2=(count_whole/count_excl)*100
                if conf1>=confidence:
                    ru1=(ele,excl,conf1)
                    ab=1
                if conf2>=confidence:
                    ru2=(excl,ele,conf2)
                    ba=1
                if ab>0:
                    rules.add(ru1)
                if ba>0:
                    rules.add(ru2)
                    
    return rules
            
        
                    