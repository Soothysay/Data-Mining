#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:36:09 2022

@author: akashchoudhuri
"""

from Template1 import temp1
from Template2 import temp2
support=50
confidence=70
op='1 and 1'
op1=list(op.split(' '))
answers=[]
if op1[0]=='1':
    code=input('CODE?')
    num=input('NUM?')
    query_ele=[]
    l=int(input('LENGTH OF QUERY?'))
    for j in range(l):
        query_ele.append(input('ENTER ELEMENT'))
    if num.isdigit():
        num=int(num)
    answer1,_=temp1(support,confidence,code,num,query_ele)
if op1[0]=='2':
    code=input('CODE?')
    num=int(input('NUM?'))
    answer1,_=temp2(support,confidence,code,num)
if op1[2]=='1':
    code=input('CODE?')
    num=input('NUM?')
    query_ele=[]
    l=int(input('LENGTH OF QUERY?'))
    for j in range(l):
        query_ele.append(input('ENTER ELEMENT'))
    if num.isdigit():
        num=int(num)
    answer2,_=temp1(support,confidence,code,num,query_ele)
if op1[2]=='2':
    code=input('CODE?')
    num=int(input('NUM?'))
    answer2,_=temp2(support,confidence,code,num)
if op1[1]=='or':
    answers=answer1
    for a2 in answer2:
        p=0
        for a in answers:
            if a2==a:
                p=1
                break
        if p==0:
            answers.append(a2)
if op1[1]=='and':
    for a2 in answer2:
        p=0
        for a1 in answer1:
            if a2==a1:
                p=1
                answers.append(a2)
                break
            

print(len(answers))
# answers is the resultant set of rules