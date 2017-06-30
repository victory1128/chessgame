#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 08:35:57 2017

@author: cnbyb
"""

import numpy as np

def new_chess_game():
    x=np.zeros([10,9])
    x[0,:]=np.array([4,3,5,6,7,6,5,3,4])
    x[3,:]=np.array([1,0,1,0,1,0,1,0,1])
    x[2,1]=2
    x[2,7]=2
    x=x-np.flipud(x)
    return np.int32(x)

def gen_x(x):
    y=np.zeros([10,9,15])
    for i in range(10):
        for j in range(9):
            if (x[i,j]!=0):
                y[i,j,x[i,j]+7]=1
    return y
def feature_map(x):
    y=np.zeros([10,9,7])
    for i in range(10):
        for j in range(9):
            if (x[i,j]<=0):
                continue
            elif (x[i,j]==7):
                if (i>0):
                    y[i-1,j,6]=1
                if (i<2):
                    y[i+1,j,6]=1
                if (j>3):
                    y[i,j-1,6]=1
                if (j<5):
                    y[i,j+1,6]=1
            elif (x[i,j]==6):
                if (i>0)&(j>3):
                    y[i-1,j-1,5]=1
                if (i>0)&(j<5):
                    y[i-1,j+1,5]=1
                if (i<2)&(j>3):
                    y[i+1,j-1,5]=1
                if (i<2)&(j<5):
                    y[i+1,j+1,5]=1
            elif (x[i,j]==5):
                if (i>0)&(j>0) and (x[i-1,j-1]==0):
                    y[i-2,j-2,4]=1
                if (i>0)&(j<8) and (x[i-1,j+1]==0):
                    y[i-2,j+2,4]=1
                if (i<4)&(j>0) and (x[i+1,j-1]==0):
                    y[i+2,j-2,4]=1
                if (i<4)&(j<8) and (x[i+1,j+1]==0):
                    y[i+2,j+2,4]=1
            elif (x[i,j]==4):
                for k in range(1,10):
                    if (i+k<=9):
                        y[i+k,j,3]=1
                    if (i+k>=9) or (x[i+k,j]!=0):
                        break
                for k in range(1,10):
                    if (i-k>=0):
                        y[i-k,j,3]=1
                    if (i-k<=0) or (x[i-k,j]!=0):
                        break
                for k in range(1,9):
                    if (j+k<=8):
                        y[i,j+k,3]=1
                    if (j+k>=8) or (x[i,j+k]!=0):
                        break
                for k in range(1,9):
                    if (j-k>=0):
                        y[i,j-k,3]=1
                    if (j-k<=0) or (x[i,j-k]!=0):
                        break
            elif (x[i,j]==3):
                if (i>1) and (j>0) and (x[i-1,j]==0):
                    y[i-2,j-1,2]=1
                if (i>0) and (j>1) and (x[i,j-1]==0):
                    y[i-1,j-2,2]=1
                if (i<8) and (j>0) and (x[i+1,j]==0):
                    y[i+2,j-1,2]=1
                if (i<9) and (j>1) and (x[i,j-1]==0):
                    y[i+1,j-2,2]=1
                if (i>1) and (j<8) and (x[i-1,j]==0):
                    y[i-2,j+1,2]=1
                if (i>0) and (j<7) and (x[i,j+1]==0):
                    y[i-1,j+2,2]=1
                if (i<8) and (j<8) and (x[i+1,j]==0):
                    y[i+2,j+1,2]=1
                if (i<9) and (j<7) and (x[i,j+1]==0):
                    y[i+1,j+2,2]=1
            elif (x[i,j]==2):
                p=0
                for k in range(1,10):
                    if (p==1) and (i+k<=9) and (x[i+k,j]==0):
                        y[i+k,j,1]=1
                    if (p==1) and (i+k<=9) and (x[i+k,j]!=0):
                        y[i+k,j,1]=1
                        break
                    if (p==0) and (i+k<=9) and (x[i+k,j]!=0):
                        p=1
                    if (i+k>=9):
                        break
                p=0
                for k in range(1,10):
                    if (p==1) and (i-k>=0) and (x[i-k,j]==0):
                        y[i-k,j,1]=1
                    if (p==1) and (i-k>=0) and (x[i-k,j]!=0):
                        y[i-k,j,1]=1
                        break
                    if (p==0) and (i+k>=0) and (x[i-k,j]!=0):
                        p=1
                    if (i-k<=0):
                        break
                p=0
                for k in range(1,9):
                    if (p==1) and (j+k<=8) and (x[i,j+k]==0):
                        y[i,j+k,1]=1
                    if (p==1) and (j+k<=8) and (x[i,j+k]!=0):
                        y[i,j+k,1]=1
                        break
                    if (p==0) and (j+k<=8) and (x[i,j+k]!=0):
                        p=1
                    if (j+k>=8):
                        break
                p=0
                for k in range(1,9):
                    if (p==1) and (j-k>=0) and (x[i,j-k]==0):
                        y[i,j-k,1]=1
                    if (p==1) and (j-k>=0) and (x[i,j-k]!=0):
                        y[i,j-k,1]=1
                        break
                    if (p==0) and (j-k>=0) and (x[i,j-k]!=0):
                        p=1
                    if (j-k<=0):
                        break
            elif (x[i,j]==1):
                if (i<9):
                    y[i+1,j,0]=1
                if (i>4) and (j>0):
                    y[i,j-1,0]=1
                if (i>4) and (j<8):
                    y[i,j+1,0]=1
    return y
def is_checkmate(x,movestepi):
    rs=x[movestepi[2],movestepi[3]]
    x[movestepi[2],movestepi[3]]=x[movestepi[0],movestepi[1]]
    x[movestepi[0],movestepi[1]]=0
    idx=np.where(x==7)
    i=idx[0][0]
    j=idx[1][0]
    p=0
    k=i+1
    while ((k<=9) and (p<2)):
        if (p==0) and ((x[k,j]==-7) or (x[k,j]==-4)):
            x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
            x[movestepi[2],movestepi[3]]=rs
            return True
        if (p==1) and (x[k,j]==-2):
            x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
            x[movestepi[2],movestepi[3]]=rs
            return True
        if (x[k,j]!=0):
            p+=1
        k+=1
    p=0
    k=i-1
    while ((k>=0) and (p<2)):
        if (p==0) and ((x[k,j]==-7) or (x[k,j]==-4)):
            x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
            x[movestepi[2],movestepi[3]]=rs
            return True
        if (p==1) and (x[k,j]==-2):
            x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
            x[movestepi[2],movestepi[3]]=rs
            return True
        if (x[k,j]!=0):
            p+=1
        k-=1
    p=0
    k=j+1
    while ((k<=8) and (p<2)):
        if (p==0) and ((x[i,k]==-7) or (x[i,k]==-4)):
            x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
            x[movestepi[2],movestepi[3]]=rs
            return True
        if (p==1) and (x[i,k]==-2):
            x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
            x[movestepi[2],movestepi[3]]=rs
            return True
        if (x[i,k]!=0):
            p+=1
        k+=1
    p=0
    k=j-1
    while ((k>=0) and (p<2)):
        if (p==0) and ((x[i,k]==-7) or (x[i,k]==-4)):
            x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
            x[movestepi[2],movestepi[3]]=rs
            return True
        if (p==1) and (x[i,k]==-2):
            x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
            x[movestepi[2],movestepi[3]]=rs
            return True
        if (x[i,k]!=0):
            p+=1
        k-=1
    if (i>0) and (((x[i-1,j-2]==-3) and (x[i-1,j-1]==0)) or ((x[i-1,j+2]==-3) and (x[i-1,j+1]==0))):
        x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
        x[movestepi[2],movestepi[3]]=rs
        return True
    if (i>1) and (((x[i-2,j-1]==-3) and (x[i-1,j-1]==0)) or ((x[i-2,j+1]==-3) and (x[i-1,j+1]==0))):
        x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
        x[movestepi[2],movestepi[3]]=rs
        return True
    if (i<9) and (((x[i+1,j-2]==-3) and (x[i+1,j-1]==0)) or ((x[i+1,j+2]==-3) and (x[i+1,j+1]==0))):
        x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
        x[movestepi[2],movestepi[3]]=rs
        return True
    if (i<8) and (((x[i+2,j-1]==-3) and (x[i+1,j-1]==0)) or ((x[i+2,j+1]==-3) and (x[i+1,j+1]==0))):
        x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
        x[movestepi[2],movestepi[3]]=rs
        return True
    if (x[i,j-1]==-1) or (x[i,j+1]==-1) or (x[i+1,j]==-1):
        x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
        x[movestepi[2],movestepi[3]]=rs
        return True
    x[movestepi[0],movestepi[1]]=x[movestepi[2],movestepi[3]]
    x[movestepi[2],movestepi[3]]=rs
    return False
    '''
    x=np.copy(x0)
    x[movestepi[2],movestepi[3]]=x[movestepi[0],movestepi[1]]
    x[movestepi[0],movestepi[1]]=0
    idx=np.where(x==7)
    i0=idx[0][0]
    j0=idx[1][0]
    for i in range(10):
        for j in range(9):
            if (x[i,j]==-7):
                if (j==j0):
                    p=0
                    for k in range(i0+1,i):
                        if (x[k,j]!=0):
                            p=1
                            break
                    if (p==0):
                        return True
            if (x[i,j]==-4):
                if (i==i0) and (j<j0):
                    p=0
                    for k in range(j+1,j0):
                        if (x[i,k]!=0):
                            p=1
                            break
                    if (p==0):
                        return True
                if (i==i0) and (j>j0):
                    p=0
                    for k in range(j0+1,j):
                        if (x[i,k]!=0):
                            p=1
                            break
                    if (p==0):
                        return True
                if (j==j0) and (i<i0):
                    p=0
                    for k in range(i+1,i0):
                        if (x[k,j]!=0):
                            p=1
                            break
                    if (p==0):
                        return True
                if (j==j0) and (i>i0):
                    p=0
                    for k in range(i0+1,i):
                        if (x[k,j]!=0):
                            p=1
                            break
                    if (p==0):
                        return True
            if (x[i,j]==-2):
                if (i==i0) and (j<j0):
                    p=0
                    for k in range(j+1,j0):
                        if (x[i,k]!=0):
                            p+=1
                    if (p==1):
                        return True
                if (i==i0) and (j>j0):
                    p=0
                    for k in range(j0+1,j):
                        if (x[i,k]!=0):
                            p+=1
                    if (p==1):
                        return True
                if (j==j0) and (i<i0):
                    p=0
                    for k in range(i+1,i0):
                        if (x[k,j]!=0):
                            p+=1
                    if (p==1):
                        return True
                if (j==j0) and (i>i0):
                    p=0
                    for k in range(i0+1,i):
                        if (x[k,j]!=0):
                            p+=1
                    if (p==1):
                        return True
            if (x[i,j]==-3):
                if (((abs(i-i0)+abs(j-j0))==3) and (i!=i0) and (j!=j0)):
                    if (i0==(i+2)) and (x[i+1,j]==0):
                        return True
                    if (i0==(i-2)) and (x[i-1,j]==0):
                        return True
                    if (j0==(j+2)) and (x[i,j+1]==0):
                        return True
                    if (j0==(j-2)) and (x[i,j-1]==0):
                        return True
    return False
    '''
def gen_next_moves(x):
    if (7 not in x) or (-7 not in x):
        return np.int32(np.zeros([0,4]))
    movestep=[]
    for i in range(10):
        for j in range(9):
            if (x[i,j]<=0):
                continue
            elif (x[i,j]==7):
                for i0 in range(i+1,10):
                    if (x[i0,j]==-7):
                        movestep.append([i,j,i0,j])
                    if (x[i0,j]!=0):
                        break
                if (i>0) and (x[i-1,j]<=0):
                    movestep.append([i,j,i-1,j])
                if (i<2) and (x[i+1,j]<=0):
                    movestep.append([i,j,i+1,j])
                if (j>3) and (x[i,j-1]<=0):
                    movestep.append([i,j,i,j-1])
                if (j<5) and (x[i,j+1]<=0):
                    movestep.append([i,j,i,j+1])
            elif (x[i,j]==6):
                if (i>0)&(j>3) and (x[i-1,j-1]<=0):
                    movestep.append([i,j,i-1,j-1])
                if (i>0)&(j<5) and (x[i-1,j+1]<=0):
                    movestep.append([i,j,i-1,j+1])
                if (i<2)&(j>3) and (x[i+1,j-1]<=0):
                    movestep.append([i,j,i+1,j-1])
                if (i<2)&(j<5) and (x[i+1,j+1]<=0):
                    movestep.append([i,j,i+1,j+1])
            elif (x[i,j]==5):
                if (i>0)&(j>0) and (x[i-2,j-2]<=0) and (x[i-1,j-1]==0):
                    movestep.append([i,j,i-2,j-2])
                if (i>0)&(j<8) and (x[i-2,j+2]<=0) and (x[i-1,j+1]==0):
                    movestep.append([i,j,i-2,j+2])
                if (i<4)&(j>0) and (x[i+2,j-2]<=0) and (x[i+1,j-1]==0):
                    movestep.append([i,j,i+2,j-2])
                if (i<4)&(j<8) and (x[i+2,j+2]<=0) and (x[i+1,j+1]==0):
                    movestep.append([i,j,i+2,j+2])
            elif (x[i,j]==4):
                for k in range(1,10):
                    if (i+k<=9) and (x[i+k,j]<=0):
                        movestep.append([i,j,i+k,j])
                    if (i+k>=9) or (x[i+k,j]!=0):
                        break
                for k in range(1,10):
                    if (i-k>=0) and (x[i-k,j]<=0):
                        movestep.append([i,j,i-k,j])
                    if (i-k<=0) or (x[i-k,j]!=0):
                        break
                for k in range(1,9):
                    if (j+k<=8) and (x[i,j+k]<=0):
                        movestep.append([i,j,i,j+k])
                    if (j+k>=8) or (x[i,j+k]!=0):
                        break
                for k in range(1,9):
                    if (j-k>=0) and (x[i,j-k]<=0):
                        movestep.append([i,j,i,j-k])
                    if (j-k<=0) or (x[i,j-k]!=0):
                        break
            elif (x[i,j]==3):
                if (i>1) and (j>0) and (x[i-1,j]==0) and (x[i-2,j-1]<=0):
                    movestep.append([i,j,i-2,j-1])
                if (i>0) and (j>1) and (x[i,j-1]==0) and (x[i-1,j-2]<=0):
                    movestep.append([i,j,i-1,j-2])
                if (i<8) and (j>0) and (x[i+1,j]==0) and (x[i+2,j-1]<=0):
                    movestep.append([i,j,i+2,j-1])
                if (i<9) and (j>1) and (x[i,j-1]==0) and (x[i+1,j-2]<=0):
                    movestep.append([i,j,i+1,j-2])
                if (i>1) and (j<8) and (x[i-1,j]==0) and (x[i-2,j+1]<=0):
                    movestep.append([i,j,i-2,j+1])
                if (i>0) and (j<7) and (x[i,j+1]==0) and (x[i-1,j+2]<=0):
                    movestep.append([i,j,i-1,j+2])
                if (i<8) and (j<8) and (x[i+1,j]==0) and (x[i+2,j+1]<=0):
                    movestep.append([i,j,i+2,j+1])
                if (i<9) and (j<7) and (x[i,j+1]==0) and (x[i+1,j+2]<=0):
                    movestep.append([i,j,i+1,j+2])
            elif (x[i,j]==2):
                p=0
                for k in range(1,10):
                    if (p==0) and (i+k<=9) and (x[i+k,j]==0):
                        movestep.append([i,j,i+k,j])
                    if (p==1) and (i+k<=9) and (x[i+k,j]!=0):
                        if (x[i+k,j]<0):
                            movestep.append([i,j,i+k,j])
                        break
                    if (p==0) and (i+k<=9) and (x[i+k,j]!=0):
                        p=1
                    if (i+k>=9):
                        break
                p=0
                for k in range(1,10):
                    if (p==0) and (i-k>=0) and (x[i-k,j]==0):
                        movestep.append([i,j,i-k,j])
                    if (p==1) and (i-k>=0) and (x[i-k,j]!=0):
                        if (x[i-k,j]<0):
                            movestep.append([i,j,i-k,j])
                        break
                    if (p==0) and (i+k>=0) and (x[i-k,j]!=0):
                        p=1
                    if (i-k<=0):
                        break
                p=0
                for k in range(1,9):
                    if (p==0) and (j+k<=8) and (x[i,j+k]==0):
                        movestep.append([i,j,i,j+k])
                    if (p==1) and (j+k<=8) and (x[i,j+k]!=0):
                        if (x[i,j+k]<0):
                            movestep.append([i,j,i,j+k])
                        break
                    if (p==0) and (j+k<=8) and (x[i,j+k]!=0):
                        p=1
                    if (j+k>=8):
                        break
                p=0
                for k in range(1,9):
                    if (p==0) and (j-k>=0) and (x[i,j-k]==0):
                        movestep.append([i,j,i,j-k])
                    if (p==1) and (j-k>=0) and (x[i,j-k]!=0):
                        if (x[i,j-k]<0):
                            movestep.append([i,j,i,j-k])
                        break
                    if (p==0) and (j-k>=0) and (x[i,j-k]!=0):
                        p=1
                    if (j-k<=0):
                        break
            elif (x[i,j]==1):
                if (i<9) and (x[i+1,j]<=0):
                    movestep.append([i,j,i+1,j])
                if (i>4) and (j>0) and (x[i,j-1]<=0):
                    movestep.append([i,j,i,j-1])
                if (i>4) and (j<8) and (x[i,j+1]<=0):
                    movestep.append([i,j,i,j+1])
    legal_movestep=[]
    for i in range(len(movestep)):
        if (x[movestep[i][2],movestep[i][3]]==-7):
            return np.array([[movestep[i][0],movestep[i][1],movestep[i][2],movestep[i][3]]])
        if (not is_checkmate(x,movestep[i])):
            #print(movestep[i])
            #print(x)
            legal_movestep.append(movestep[i])
    if len(legal_movestep)>0:
        return np.array(legal_movestep)
    else:
        return np.int32(np.zeros([0,4]))

if (__name__=='__main__'):
    x=new_chess_game()
    movestep=gen_next_moves(x)