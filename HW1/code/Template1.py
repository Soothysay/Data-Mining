#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:48:50 2022

@author: akashchoudhuri
"""
from itertools import combinations
from rulegen import rg
code='HEAD'
num=1
query_ele=['gene59_up','gene10_down']

#query_ele=['gene59_up']

def temp1(support,confidence,code,num,query_ele):
    rules=rg(support,confidence)
    bodys=[]
    heads=[]
    rules1=[]
    rule_answer=[]
    for r in rules:
        unit=list()
        unit.append(list(r[0]))
        unit.append(list(r[1]))
        rules1.append(unit)
    for r in rules1:
        heads.append(r[1])
        bodys.append(r[0])
    if code=='HEAD':
        if num=='ANY':
            found=set()
            for it in query_ele:
                for h in range(len(heads)):
                    for h1 in range(len(heads[h])):
                        if heads[h][h1]==it:
                            found.add(h)
        elif num=='NONE':
            found=set()
            for it in query_ele:
                for h in range(len(heads)):
                    for h1 in range(len(heads[h])):
                        if heads[h][h1]==it:
                            found.add(h)
            whole=set()
            #counter1=len(rules1)-counter
            for i in range(len(rules1)):
                whole.add(i)
            notf=whole-found
            found=notf
        elif type(num)==int:
            len_query=len(query_ele)
            p=0
            combi=[]
            if num<len_query:
                p=1
                for i in range(num+1,len_query+1):
                    x=list(combinations(query_ele,i))
                    for j in x:
                        combi.append(j)
            elif num>len_query:
                print('Exit')
                return ['Bad Input'],0
            finder_comb=list(combinations(query_ele,num))
            found=set()
            remover=set()
            for h in range(len(heads)):
                for it in finder_comb:
                    if set(it).issubset(heads[h]):
                        #print(heads[h])
                        found.add(h)
                if p==1:
                    for it1 in combi:
                        if set(it1).issubset(heads[h]):
                            remover.add(h)
            found=found-remover
            
    
    elif code=='BODY':
        if num=='ANY':
            found=set()
            for it in query_ele:
                for h in range(len(bodys)):
                    for h1 in range(len(bodys[h])):
                        if bodys[h][h1]==it:
                            found.add(h)
        elif num=='NONE':
            found=set()
            for it in query_ele:
                for h in range(len(bodys)):
                    for h1 in range(len(bodys[h])):
                        if bodys[h][h1]==it:
                            found.add(h)
            whole=set()
            #counter1=len(rules1)-counter
            for i in range(len(rules1)):
                whole.add(i)
            notf=whole-found
            found=notf
            
        elif type(num)==int:
            len_query=len(query_ele)
            p=0
            combi=[]
            if num<len_query:
                p=1
                for i in range(num+1,len_query+1):
                    x=list(combinations(query_ele,i))
                    for j in x:
                        combi.append(j)
            elif num>len_query:
                print('Exit')
            finder_comb=list(combinations(query_ele,num))
            found=set()
            remover=set()
            for h in range(len(bodys)):
                for it in finder_comb:
                    if set(it).issubset(bodys[h]):
                        #print(heads[h])
                        found.add(h)
                if p==1:
                    for it1 in combi:
                        if set(it1).issubset(heads[h]):
                            remover.add(h)
            found=found-remover
    
    elif code=='RULE':
        if num=='ANY':
            found=set()
            for it in query_ele:
                for h in range(len(rules1)):
                    for h1 in range(len(rules1[h])):
                        for h2 in range(len(rules1[h][h1])):
                            if rules1[h][h1][h2]==it:
                                found.add(h)
        elif num=='NONE':
            found=set()
            for it in query_ele:
                for h in range(len(rules1)):
                    for h1 in range(len(rules1[h])):
                        for h2 in range(len(rules1[h][h1])):
                            if rules1[h][h1][h2]==it:
                                found.add(h)
            whole=set()
            #counter1=len(rules1)-counter
            for i in range(len(rules1)):
                whole.add(i)
            notf=whole-found
            found=notf
            
        elif type(num)==int:
            len_query=len(query_ele)
            p=0
            combi=[]
            if num<len_query:
                p=1
                for i in range(num+1,len_query+1):
                    x=list(combinations(query_ele,i))
                    for j in x:
                        combi.append(j)
            elif num>len_query:
                print('Exit')
            finder_comb=list(combinations(query_ele,num))
            found=set()
            remover=set()
            for h in range(len(rules1)):
                stuff=set()
                for h1 in range(len(rules1[h])):
                    for h2 in range(len(rules1[h][h1])):
                        stuff.add(rules1[h][h1][h2])
            
                for it in finder_comb:
                    if set(it).issubset(stuff):
                        #print(heads[h])
                        found.add(h)
                if p==1:
                    for it1 in combi:
                        if set(it1).issubset(stuff):
                            remover.add(h)
            found=found-remover
    for i in found:
        rule_answer.append(rules1[i])
        
    return rule_answer,len(rule_answer)

answer,count=temp1(50,70,code,num,query_ele)
print(count)