#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:59:50 2017

@author: cnbyb
"""

import ChessStep
import numpy as np
import json
#import os
with open('item_new.json','r') as f:
    data=json.load(f)
with open('item_new2.json','r') as f:
    data2=json.load(f)
data=data+data2
Xchess=[]
Xstep=[]
Xresult=[]
for i in range(len(data)):
    if (i%1000==0):
        print(i,'==')
    datai=data[i]
    result=0
    if ('WIN' in datai['conclusion']):
        result=1
    elif ('LOSS' in datai['conclusion']):
        result=-1
    movei=datai['move']
    x=ChessStep.new_chess_game()
    rd=0
    if (len(movei)%4!=0):
        print(i)
        continue
    if (not movei.isdigit()):
        print(i)
        continue
    p_append=True
    Xchessi=[]
    Xstepi=[]
    Xresulti=[]
    for j in range(int(len(movei)/4.0)):
        i1=abs(rd-int(movei[j*4+1]))
        j1=int(movei[j*4])
        i2=abs(rd-int(movei[j*4+3]))
        j2=int(movei[j*4+2])
        moveij=ChessStep.gen_next_moves(x)
        if ([i1,j1,i2,j2] in moveij.tolist()):
            Xchessi.append(np.copy(x))
            Xstepi.append(np.array([i1,j1,i2,j2]))
            Xresulti.append(np.copy(result))
        else:
            print(i,j)
            p_append=False
            break
        x[i2,j2]=x[i1,j1]
        x[i1,j1]=0
        x=-np.flipud(x)
        rd=abs(rd-9)
        result=-result
    if (p_append):
        for j in range(len(Xchessi)):
            Xchess.append(np.copy(Xchessi[j]))
            Xstep.append(np.copy(Xstepi[j]))
            Xresult.append(np.copy(Xresulti[j]))
Xchess=np.stack(Xchess,axis=0)
Xstep=np.stack(Xstep,axis=0)
np.savez('SLdata_new',Xchess,Xstep,np.array(Xresult))
