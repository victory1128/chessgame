#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:36:13 2017

@author: cnbyb
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
def Multi_attention(x,head_num,fin,fout):
    xsize=x.get_shape().as_list()
    batch_size=xsize[0]
    for i in range(head_num):
        with tf.variable_scope('attention_'+str(i)):
            with tf.variable_scope('q'):
                qi=cnn_layer(x,1,fin,fout)
            with tf.variable_scope('k'):
                ki=cnn_layer(x,1,fin,fout)
            with tf.variable_scope('v'):
                vi=cnn_layer(x,1,fin,fout)
            att=tf.nn.softmax(tf.matmul(tf.reshape(qi,[batch_size,-1,fout]),tf.transpose(tf.reshape(ki,[batch_size,-1,fout]),[0,2,1]))/tf.cast(tf.constant(np.sqrt(fout)),tf.float32))
            if (i==0):
                y=tf.reshape(tf.matmul(att,tf.reshape(vi,[batch_size,-1,fout])),[batch_size,10,9,-1])
            else:
                y=tf.concat([y,tf.reshape(tf.matmul(att,tf.reshape(vi,[batch_size,-1,fout])),[batch_size,10,9,-1])],3)
    return y
def Model(x,y_,cy,T,keep_prob,is_training):
    xsize=x.get_shape().as_list()
    batch_size=xsize[0]
    fin=xsize[-1]
    fout_per_head=8
    head_num=8
    fout=fout_per_head*head_num
    position=fin
    b=tf.get_variable(name='position_embedding',shape=[10,9,position],dtype='float')
    with tf.variable_scope('Policy'):
        tf.add_to_collection('ema_op',tf.no_op())
        with tf.variable_scope('pre_block'):
            with tf.variable_scope('pre1'):
                y=block(tf.concat([x,x*b,0*x+b],3),5,fout)
            y,ema=bn(y,fout,is_training,True)
            tf.add_to_collection('ema_op',ema)
            with tf.variable_scope('pre_att'):
                y=Multi_attention(y,head_num,fout,fout_per_head)+y
            y=tf.nn.dropout(y,keep_prob)
        with tf.variable_scope('block_0'):
            y=block(y,3,fout)
        with tf.variable_scope('block_1'):
            y=block(y,3,fout)
            y,ema=bn(y,fout,is_training,True)
            tf.add_to_collection('ema_op',ema)
        with tf.variable_scope('att'):
            y=Multi_attention(y,head_num,fout,fout_per_head)+y
        with tf.variable_scope('output'):
            y=tf.nn.dropout(y,keep_prob)
            with tf.variable_scope('block_1'):
                with tf.variable_scope('y1_0'):
                    y1=block(y,3,fout)
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
            tf.add_to_collection('ac',tf.cast(tf.equal(tf.argmax(p,0),tf.cast(cy[i],tf.int64)),'float'))
            tf.add_to_collection(name='losses',value=-tf.log(p[cy[i]]+1e-5))
            batch_p.append(p)
        loss=tf.add_n(tf.get_collection('losses'),name='loss')/2
        acr=tf.add_n(tf.get_collection('ac'),name='acr')/2
    return batch_p,loss,acr,tf.get_collection('ema_op')

class SLdata:
    def __init__(self):
        r=np.load('SLdata_new.npz')
        self.Xchess=r['arr_0']
        self.Xstep=r['arr_1']
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
        X=np.zeros([batch_size,10,9,29])
        Xs=np.int32(np.zeros([batch_size,120,4]))
        T=np.int32(np.zeros([batch_size]))
        cy=np.int32(np.zeros([batch_size]))
        for i in range(batch_size):
            #print(i)
            Xi=self.Xchess[jr[i],:,:]
            X[i,:,:,0:15]=ChessStep.gen_x(Xi)
            X[i,:,:,15:22]=ChessStep.feature_map(Xi)
            X[i,:,:,22:29]=-np.flipud(ChessStep.feature_map(-np.flipud(Xi)))
            movei=ChessStep.gen_next_moves(Xi)
            m=np.shape(movei)[0]
            Xs[i,0:m,:]=movei
            T[i]=m
            moveilist=movei.tolist()
            cy[i]=moveilist.index(self.Xstep[jr[i],:].tolist())
        return X,Xs,T,cy
if (__name__=='__main__'):
    X=SLdata()
    batch_size=200
    x=tf.placeholder('float',[batch_size,10,9,29])
    y_=tf.placeholder(tf.int32,[batch_size,120,4])
    T=tf.placeholder(tf.int32,[batch_size])
    cy=tf.placeholder(tf.int32,[batch_size])
    keep_prob=tf.placeholder(tf.float32)
    is_training=tf.placeholder(tf.bool)
    batch_p,loss,acr,control_op=Model(x,y_,cy,T,keep_prob,is_training)
    print('network established')
    global_step=tf.Variable(0,trainable=False)
    lr=tf.train.exponential_decay(0.5,global_step,10000,0.8,staircase=True)
    #lr=0.5
    control_op.append(tf.assign_add(global_step,1))
    with tf.control_dependencies(control_op):
        train_op=tf.train.AdadeltaOptimizer(lr).minimize(loss)
    vlist=tf.global_variables()
    saver_dict={}
    for i in range(len(vlist)):
        vname=vlist[i].name
        if ('Policy' in vname) and ('Adadelta' not in vname):
            saver_dict[vname]=vlist[i]
            print(vname)
    saver=tf.train.Saver(saver_dict)
    saver2=tf.train.Saver()
    tf_config=tf.ConfigProto(allow_soft_placement=True)  
    tf_config.gpu_options.per_process_gpu_memory_fraction=0.4
    sess=tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,'./rl0.ckpt')
    for i in range(100001):
        X_batch,Xs_batch,T_batch,cy_batch=X.mini_batch(batch_size,is_tr=True,is_shuffle=True)
        #print(1)
        sess.run(train_op,feed_dict={x:X_batch,y_:Xs_batch,T:T_batch,cy:cy_batch,keep_prob:0.75,is_training:True})
        #print(2)
        if (i%10==0):
            X_batch,Xs_batch,T_batch,cy_batch=X.mini_batch(batch_size,is_tr=True,is_shuffle=True)
            print(i,sess.run([loss,acr],feed_dict={x:X_batch,y_:Xs_batch,T:T_batch,cy:cy_batch,keep_prob:1.0,is_training:False}))
        if (i%1000==0):
            save_path=saver.save(sess,'rl0_new.ckpt')
            save_path2=saver2.save(sess,'SL_new.ckpt')
            ac_sum=0.0
            batch_num=int(10000/batch_size)
            for j in range(batch_num):
                X_batch,Xs_batch,T_batch,cy_batch=X.mini_batch(batch_size,is_tr=False,is_shuffle=False,st=j*batch_size)
                ac_sum+=sess.run(acr,feed_dict={x:X_batch,y_:Xs_batch,T:T_batch,cy:cy_batch,keep_prob:1.0,is_training:False})
            print(i,ac_sum/batch_num)