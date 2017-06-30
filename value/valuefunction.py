#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:34:38 2017

@author: ruos

"""
import tensorflow as tf
import numpy as np
import random
import ChessStep

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


# 1000, 0.8835, 0.5626
def Model1(x,y_,keep_prob,is_training):
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
        y=tf.nn.dropout(y,keep_prob)
    with tf.variable_scope('block_1'):
        y=block(y,3,fout)
        y,ema=bn(y,fout,is_training,True)
        tf.add_to_collection('ema_op',ema)
    with tf.variable_scope('output'):
        y=tf.nn.dropout(y,keep_prob)
        with tf.variable_scope('block_1'):
            with tf.variable_scope('y1_0'):
                y=block(y,3,fout)
                
    y=tf.reshape(y,[batch_size,-1])
    w=tf.get_variable(name='pw',shape=[90*fout,1],dtype='float')
    y=tf.sigmoid(tf.matmul(y,w))
    loss=tf.reduce_mean(-y_*tf.log(y[:,0]+1e-10)-(1-y_)*tf.log(1-y[:,0]+1e-10))
    return y[:,0],loss,tf.get_collection('ema_op')


def Model2(x,y_,keep_prob,is_training):
    xsize=x.get_shape().as_list()
    batch_size=xsize[0]
    l = 5
    fin=xsize[-1]-1+l
    fout=64
    position=fin
    piece=tf.get_variable('piece_embedding',shape=[15,l],dtype='float')
    y=tf.concat([tf.nn.embedding_lookup(piece,tf.cast(x[:,:,:,0],tf.int32)),x[:,:,:,1:xsize[-1]]],3)
    b=tf.get_variable(name='position_embedding',shape=[10, 9, position],dtype='float')
    with tf.variable_scope('Policy'):
        tf.add_to_collection('ema_op',tf.no_op())
    with tf.variable_scope('pre_block'):
        with tf.variable_scope('pre1'):
            y=block(tf.concat([y,y*b,0*y+b],3),5,fout)
        y,ema=bn(y,fout,is_training,True)
        tf.add_to_collection('ema_op',ema)
        y=tf.nn.dropout(y,keep_prob)
    with tf.variable_scope('block_1'):
        y=block(y,3,fout)
        y,ema=bn(y,fout,is_training,True)
        tf.add_to_collection('ema_op',ema)
    with tf.variable_scope('output'):
        y=tf.nn.dropout(y,keep_prob)
        with tf.variable_scope('block_1'):
            with tf.variable_scope('y1_0'):
                y=block(y,1,fout)
                
    y=tf.reshape(y,[batch_size,-1])
    w=tf.get_variable(name='pw',shape=[90*fout,1],dtype='float')
    b=tf.get_variable(name='b', shape=[batch_size,1])
    y=tf.sigmoid(tf.matmul(y,w)+b)
    loss=tf.reduce_mean(tf.square(y_ - y[:,0]))
    return y[:,0],loss,tf.get_collection('ema_op')


class SLdata:
    def __init__(self):
        r=np.load('SLdata_new.npz')
        self.Xchess=r['arr_0']
        self.ychess=r['arr_2']
        #self.Xresult=['arr_2']
        self.N=np.shape(self.Xchess)[0]
        self.tr=2000000
    def mini_batch(self,batch_size,is_tr=True,is_shuffle=True,st=None):
        if (is_tr):
            J=list(range(self.tr))
        else:
            J=list(range(self.tr,self.N))
        if (is_shuffle):
            jr=random.sample(J,batch_size)
        else:
            jr=J[st:(st+batch_size)]
        X=np.zeros([batch_size,10,9,15])
        y=np.zeros([batch_size])
        for i in range(batch_size):
            #print(i)
            Xi=self.Xchess[jr[i],:,:]
            X[i,:,:,0]=Xi+7
            X[i,:,:,1:8]=ChessStep.feature_map(Xi)
            X[i,:,:,8:15]=-np.flipud(ChessStep.feature_map(-np.flipud(Xi)))
            y[i]=(1.0+self.ychess[jr[i]])/2
        return X,y
if (__name__=='__main__'):
    X=SLdata()
    batch_size=100
    x=tf.placeholder('float',[batch_size,10,9,15])
    y_=tf.placeholder(tf.float32,[batch_size])
    keep_prob=tf.placeholder(tf.float32)
    is_training=tf.placeholder(tf.bool)
    batch_q,loss,control_op=Model2(x,y_,keep_prob,is_training)
    print('network established')
    global_step=tf.Variable(0,trainable=False)
    lr=tf.train.exponential_decay(1e-3,global_step,10000,0.8,staircase=True)
    #lr=0.5
    control_op.append(tf.assign_add(global_step,1))
    with tf.control_dependencies(control_op):
        train_op=tf.train.RMSPropOptimizer(lr).minimize(loss)
    saver2=tf.train.Saver()
    tf_config=tf.ConfigProto(allow_soft_placement=True)  
    tf_config.gpu_options.per_process_gpu_memory_fraction=0.1
    sess=tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    #saver2.restore(sess,'./QNET.ckpt')
    #62
    for i in range(1,10001):
        X_batch,y_batch=X.mini_batch(batch_size,is_tr=True,is_shuffle=True)
        #print(1)
        sess.run(train_op,feed_dict={x:X_batch,y_:y_batch,keep_prob:0.75,is_training:True})
        #print(2)
        if (i%10==0):
            X_batch,y_batch=X.mini_batch(batch_size,is_tr=True,is_shuffle=True)
            print(i,sess.run([loss],feed_dict={x:X_batch,y_:y_batch,keep_prob:1.0,is_training:False}))
        if (i%1000==0):
            save_path2=saver2.save(sess,'QNET.ckpt')
            loss_sum=0.0
            batch_num=int(10000/batch_size)
            totalnum=0.0
            acnum=0.0
            for j in range(batch_num):
                X_batch,y_batch=X.mini_batch(batch_size,is_tr=False,is_shuffle=False,st=j*batch_size)
                loss_sum+=sess.run(loss,feed_dict={x:X_batch,y_:y_batch,keep_prob:1.0,is_training:False})
                qvalue=sess.run(batch_q,feed_dict={x:X_batch,y_:y_batch,keep_prob:1.0,is_training:False})
                totalnum+=np.sum(np.abs(y_batch*2-1.0)>1e-3)
                acnum+=np.sum((y_batch*2-1.0)*(qvalue*2-1.0)>1e-3)
            print(i,loss_sum/batch_num,acnum/totalnum)