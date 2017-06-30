#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YongfengLi
"""

import tensorflow as tf
import numpy as np
import random
import os

def bn(x,size,is_training,is_cnn_layer):
    ema=tf.train.ExponentialMovingAverage(0.999)
    bias=tf.get_variable(name='bn_bias',initializer=tf.zeros(size))
    gamma=tf.get_variable(name='bn_gamma',initializer=tf.ones(size))
    e=1e-5
    if (is_cnn_layer):
        axes=[0,1,2]
    else:
        axes=[0]
    mean,variance=tf.nn.moments(x,axes)
    ema_op=ema.apply([mean,variance])
    m=tf.cond(is_training,lambda:mean,lambda:ema.average(mean))
    v=tf.cond(is_training,lambda:variance,lambda:ema.average(variance))
    xbn=tf.nn.batch_normalization(x,m,v,bias,gamma,e)
    return xbn,ema_op

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def cnn_layer(x,size,fin,fout):
    w=tf.get_variable(name='sq',shape=[size,size,fin,fout],dtype='float')
    b=tf.get_variable(name='sqb',shape=[fout],dtype='float')
    y=conv2d(x,w)+b
    return tf.nn.relu(y)

def block(x,size,fout):
    fin=x.get_shape().as_list()[-1]
    with tf.variable_scope('layer1'):
        y=cnn_layer(x,size,fin,fout)
    with tf.variable_scope('layer2'):
        y=cnn_layer(y,size,fout,fout)
    if (fin==fout):
        y=y+x
    elif (fin<fout):
        xsize=x.get_shape().as_list()
        y=y+tf.concat([x,tf.zeros([xsize[0],xsize[1],xsize[2],fout-fin])],3)
    else:
        with tf.variable_scope('res_proj'):
            w=tf.get_variable(name='res_proj_w',shape=[1,1,fin,fout],dtype='float')
            y=y+conv2d(x,w)
    return y

def Model(x,y_,cy,T,keep_prob,is_training):
    xsize=x.get_shape().as_list()
    batch_size=xsize[0]
    fin=xsize[-1]-1+16
    fout=64
    position=fin
    piece=tf.get_variable('piece_embedding',shape=[15,16],dtype='float')
    y=tf.concat([tf.nn.embedding_lookup(piece,tf.cast(x[:,:,:,0],tf.int32)),x[:,:,:,1:xsize[-1]]],3)
    b=tf.get_variable(name='position_embedding',shape=[10,9,position],dtype='float')
    with tf.variable_scope('Policy'):
        tf.add_to_collection('ema_op',tf.no_op())
        with tf.variable_scope('pre_block'):
            with tf.variable_scope('pre1'):
                y=block(tf.concat([y,y*b,0*y+b],3),5,fout)
            y,ema=bn(y,fout,is_training,True)
            tf.add_to_collection('ema_op',ema)
        with tf.variable_scope('output'):
            y=tf.nn.dropout(y,keep_prob)
            with tf.variable_scope('block_1'):
                with tf.variable_scope('y1'):
                    y1=y
                #b1=tf.get_variable(name='bias_1',initializer=tf.zeros([90,1]))
                #b2=tf.get_variable(name='bias_2',initializer=tf.zeros([90,1]))
        w=tf.get_variable(name='pw',shape=[2*fout,1],dtype='float')
        batch_p=[]
        for i in range(batch_size):
            #print(i)
            y1id=y_[i,0:T[i],0]*9+y_[i,0:T[i],1]
            y2id=y_[i,0:T[i],2]*9+y_[i,0:T[i],3]
            y1_lookup=tf.nn.embedding_lookup(tf.reshape(y1[i,:,:,:],[90,-1]),y1id)
            y2_lookup=tf.nn.embedding_lookup(tf.reshape(y1[i,:,:,:],[90,-1]),y2id)
            #b1_lookup=tf.nn.embedding_lookup(b1,y1id)
            #b2_lookup=tf.nn.embedding_lookup(b2,y2id)
            p=tf.matmul(tf.concat([y1_lookup,y2_lookup],1),w)[:,0]#+b1_lookup[:,0]+b2_lookup[:,0]
            p=tf.nn.softmax(p)
            tf.add_to_collection('ac_r',tf.cast(tf.equal(tf.argmax(p,0),tf.cast(cy[i],tf.int64)),'float'))
            batch_p.append(p)
    return batch_p

def Model2(x,y_,cy,T,keep_prob,is_training):
    xsize=x.get_shape().as_list()
    batch_size=xsize[0]
    fin=xsize[-1]-1+16
    fout=64
    position=fin
    piece=tf.get_variable('piece_embedding',shape=[15,16],dtype='float')
    y=tf.concat([tf.nn.embedding_lookup(piece,tf.cast(x[:,:,:,0],tf.int32)),x[:,:,:,1:xsize[-1]]],3)
    b=tf.get_variable(name='position_embedding',shape=[10,9,position],dtype='float')
    with tf.variable_scope('Policy'):
        tf.add_to_collection('ema_op',tf.no_op())
        with tf.variable_scope('pre_block'):
            with tf.variable_scope('pre1'):
                y=block(tf.concat([y,y*b,0*y+b],3),5,fout)
            y,ema=bn(y,fout,is_training,True)
            tf.add_to_collection('ema_op',ema)
            with tf.variable_scope('pre_att'):
                yatt=block(y,5,fout)
            att1=tf.nn.softmax(yatt,dim=1)
            att2=tf.nn.softmax(yatt,dim=2)
            y=tf.concat([y,y*att1,y*att2],3)
            y=tf.nn.dropout(y,keep_prob)
        with tf.variable_scope('block_0'):
            y=block(y,3,fout)
        with tf.variable_scope('block_1'):
            y=block(y,3,fout)
            y,ema=bn(y,fout,is_training,True)
            tf.add_to_collection('ema_op',ema)
        with tf.variable_scope('att'):
            yatt=block(y,3,fout)
            att1=tf.nn.softmax(yatt,dim=1)
            att2=tf.nn.softmax(yatt,dim=2)
            y=tf.concat([y,y*att1,y*att2],3)
        with tf.variable_scope('output'):
            y=tf.nn.dropout(y,keep_prob)
            with tf.variable_scope('block_1'):
                with tf.variable_scope('y1_0'):
                    y=block(y,3,fout)
                with tf.variable_scope('y1'):
                    y1=block(y,1,fout)
                #b1=tf.get_variable(name='bias_1',initializer=tf.zeros([90,1]))
            with tf.variable_scope('block_2'):
                y=tf.nn.dropout(y,keep_prob)
                with tf.variable_scope('y2_0'):
                    y=block(y,3,fout)
                with tf.variable_scope('y2'):
                    y2=block(y,1,fout)
                #b2=tf.get_variable(name='bias_2',initializer=tf.zeros([90,1]))
        w=tf.get_variable(name='pw',shape=[4*fout,1],dtype='float')
        batch_p=[]
        for i in range(batch_size):
            #print(i)
            y1id=y_[i,0:T[i],0]*9+y_[i,0:T[i],1]
            y2id=y_[i,0:T[i],2]*9+y_[i,0:T[i],3]
            y1_lookup=tf.nn.embedding_lookup(tf.reshape(y1[i,:,:,:],[90,-1]),y1id)
            y2_lookup=tf.nn.embedding_lookup(tf.reshape(y2[i,:,:,:],[90,-1]),y2id)
            #b1_lookup=tf.nn.embedding_lookup(b1,y1id)
            #b2_lookup=tf.nn.embedding_lookup(b2,y2id)
            p=tf.matmul(tf.concat([y1_lookup,y2_lookup,y1_lookup-y2_lookup,y1_lookup*y2_lookup],1),w)[:,0]#+b1_lookup[:,0]+b2_lookup[:,0]
            p=tf.nn.softmax(p)
            tf.add_to_collection('ac_s',tf.cast(tf.equal(tf.argmax(p,0),tf.cast(cy[i],tf.int64)),'float'))
            batch_p.append(p)
    return batch_p

def Model3(x,keep_prob,is_training):
    xsize=x.get_shape().as_list()
    batch_size=xsize[0]
    fin=xsize[-1]-1+16
    fout=64
    position=fin
    piece=tf.get_variable('piece_embedding',shape=[15,16],dtype='float')
    y=tf.concat([tf.nn.embedding_lookup(piece,tf.cast(x[:,:,:,0],tf.int32)),x[:,:,:,1:xsize[-1]]],3)
    b=tf.get_variable(name='position_embedding',shape=[10,9,position],dtype='float')
    with tf.variable_scope('Policy'):
        tf.add_to_collection('ema_op',tf.no_op())
        with tf.variable_scope('pre_block'):
            with tf.variable_scope('pre1'):
                y=block(tf.concat([y,y*b,0*y+b],3),5,fout)
            y,ema=bn(y,fout,is_training,True)
            tf.add_to_collection('ema_op',ema)
            with tf.variable_scope('pre_att'):
                yatt=block(y,5,fout)
            att1=tf.nn.softmax(yatt,dim=1)
            att2=tf.nn.softmax(yatt,dim=2)
            y=tf.concat([y,y*att1,y*att2],3)
            y=tf.nn.dropout(y,keep_prob)
        with tf.variable_scope('block_0'):
            y=block(y,3,fout)
        with tf.variable_scope('block_1'):
            y=block(y,3,fout)
            y,ema=bn(y,fout,is_training,True)
            tf.add_to_collection('ema_op',ema)
        with tf.variable_scope('att'):
            yatt=block(y,3,fout)
            att1=tf.nn.softmax(yatt,dim=1)
            att2=tf.nn.softmax(yatt,dim=2)
            y=tf.concat([y,y*att1,y*att2],3)
        with tf.variable_scope('output'):
            y=tf.nn.dropout(y,keep_prob)
            with tf.variable_scope('block_1'):
                with tf.variable_scope('y1_0'):
                    y=block(y,3,fout)
                #b1=tf.get_variable(name='bias_1',initializer=tf.zeros([90,1]))
            with tf.variable_scope('block_2'):
                y=tf.nn.dropout(y,keep_prob)
                with tf.variable_scope('y2_0'):
                    y=block(y,3,fout)
                y,ema=bn(y,fout,is_training,True)
                tf.add_to_collection('ema_op',ema)
                with tf.variable_scope('y2'):
                    y=block(y,1,fout)
                #b2=tf.get_variable(name='bias_2',initializer=tf.zeros([90,1]))
        y=tf.reshape(y,[batch_size,-1])
        w=tf.get_variable(name='pw',shape=[90*fout,1],dtype='float')
        y=tf.sigmoid(tf.matmul(y,w))
        #loss=tf.reduce_mean(-y_*tf.log(y[:,0]+1e-10)-(1-y_)*tf.log(1-y[:,0]+1e-10))
    return y[:,0]#,loss,tf.get_collection('ema_op')


