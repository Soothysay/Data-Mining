#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 13:34:51 2022

@author: akashchoudhuri
"""

from rulegen import rg
code='HEAD'
num=1
def temp2(support,confidence,code,num):

    rules=rg(support,confidence)
    bodys=[]
    heads=[]
    rules1=[]
    rule_take=set()
    for r in rules:
        unit=list()
        unit.append(list(r[0]))
        unit.append(list(r[1]))
        rules1.append(unit)
    for r in rules1:
        heads.append(r[1])
        bodys.append(r[0])
    found=set()
    if code=='HEAD':
        for h in range(len(heads)):
            if len(heads[h])>=num:
                found.add(h)
    if code=='BODY':
        for h in range(len(bodys)):
            if len(bodys[h])>=num:
                found.add(h)
    if code=='RULE':
        for h in range(len(bodys)):
            if len(heads[h])+len(bodys[h])>=num:
                found.add(h)
    
    rule_answer=[]
    for i in found:
        rule_answer.append(rules1[i])
        
    return rule_answer,len(rule_answer)

answer,count=temp2(50,70,code,num)
print(count)