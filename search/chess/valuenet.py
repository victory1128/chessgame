#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YongfengLi
"""

import tensorflow as tf
import numpy as np
import random

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


def Model2(x,keep_prob,is_training):
    xsize=x.get_shape().as_list()
    batch_size=xsize[0]
    l = 2
    fin=xsize[-1]-1+l
    fout=64
    position=fin
    piece=tf.get_variable('piece_embedding',shape=[15,l],dtype='float')
    y=tf.concat([tf.nn.embedding_lookup(piece,tf.cast(x[:,:,:,0],tf.int32)),x[:,:,:,1:xsize[-1]]],3)
    b=tf.get_variable(name='position_embedding',shape=[10,9,position],dtype='float')
    with tf.variable_scope('Policy'):
        tf.add_to_collection('ema_op',tf.no_op())
    with tf.variable_scope('pre_block'):
        with tf.variable_scope('pre1'):
            y=block(tf.concat([y,y*b,0*y+b],3),5,fout)
        y,ema=bn(y,fout,is_training,True)
       # tf.add_to_collection('ema_op',ema)
        y=tf.nn.dropout(y,keep_prob)
    with tf.variable_scope('block_1'):
        y=block(y,3,fout)
        y,ema=bn(y,fout,is_training,True)
       # tf.add_to_collection('ema_op',ema)
    with tf.variable_scope('output'):
        y=tf.nn.dropout(y,keep_prob)
        with tf.variable_scope('block_1'):
            with tf.variable_scope('y1_0'):
                y=block(y,1,fout)
                
    y=tf.reshape(y,[batch_size,-1])
    w=tf.get_variable(name='pw',shape=[90*fout,1],dtype='float')
    y=tf.sigmoid(tf.matmul(y,w))
    #loss=tf.reduce_mean(-y_*tf.log(y[:,0]+1e-10)-(1-y_)*tf.log(1-y[:,0]+1e-10))
    return y[:,0]#,loss,tf.get_collection('ema_op')


