#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YongfengLi
"""
 
from . import ChessStep2 as cs    
from . import network as nw   
import tensorflow as tf
import numpy as np

class policy_nn(object):
    
    def __init__(self):
        batch_size=1
        self.x = tf.placeholder('float',[batch_size,10,9,15])
        self.y_ = tf.placeholder(tf.int32,[batch_size,None,4])
        self.T = tf.placeholder(tf.int32,[batch_size])
        self.cy = tf.placeholder(tf.int32,[batch_size])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        
        with tf.variable_scope('SLPREDICT'):
            self.sl_p = nw.Model2(self.x, self.y_, self.cy, self.T, self.keep_prob, self.is_training)
        vlist=tf.global_variables()
        
        SLlist={}
        for i in range(len(vlist)):
            vname=vlist[i].name
            if ('SLPREDICT' in vname) and ('Adadelta' not in vname):
                SLlist[vname.replace('SLPREDICT/','').replace(':0','')]=vlist[i]#vname.replace('SLPREDICT/','')
        
        saver2=tf.train.Saver(SLlist)
        self.sess=tf.Session()
        saver2.restore(self.sess,'./param/sl_emb.ckpt')
    
    def __call__(self, pos):
        Chess_batch = np.zeros([1,10,9,15])
        Chess_batch[0,:,:,0] = pos+7
        Chess_batch[0,:,:,1:8] = cs.feature_map(pos)
        Chess_batch[0,:,:,8:15] = -np.flipud(cs.feature_map(-np.flipud(pos)))
        movei = cs.gen_next_moves(pos)
        mm = np.shape(movei)[0]
        Xs_batch = np.zeros([1,mm,4],dtype='int32')
        Xs_batch[0,0:mm,:] = movei
        T_batch = np.zeros([1])
        T_batch[0] = mm
        if T_batch[0] > 0:
            action_p = self.sess.run(self.sl_p, feed_dict = { self.x:Chess_batch, self.y_:Xs_batch, self.T:T_batch, self.cy:T_batch, self.keep_prob:1, self.is_training:False } )
            return movei, action_p
        else:
            return movei, 0


class rollout_nn(object):
    
    def __init__(self):
        batch_size=1
        self.x = tf.placeholder('float',[batch_size,10,9,15])
        self.y_ = tf.placeholder(tf.int32,[batch_size,None,4])
        self.T = tf.placeholder(tf.int32,[batch_size])
        self.cy = tf.placeholder(tf.int32,[batch_size])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        
        with tf.variable_scope('ROLLOUT'):
            self.rollout_p = nw.Model(self.x, self.y_, self.cy, self.T, self.keep_prob, self.is_training)
        vlist=tf.global_variables()
        
        ROlist={}
        for i in range(len(vlist)):
            vname=vlist[i].name
            if ('ROLLOUT' in vname) and ('Adadelta' not in vname):
                ROlist[vname.replace('ROLLOUT/','').replace(':0','')]=vlist[i]

        saver=tf.train.Saver(ROlist)
        self.sess=tf.Session()
        saver.restore(self.sess,'./param/rollout.ckpt')
    
    def __call__(self, pos):
        Chess_batch = np.zeros([1,10,9,15])
        Chess_batch[0,:,:,0] = pos+7
        Chess_batch[0,:,:,1:8] = cs.feature_map(pos)
        Chess_batch[0,:,:,8:15] = -np.flipud(cs.feature_map(-np.flipud(pos)))
        movei = cs.gen_next_moves(pos)
        mm = np.shape(movei)[0]
        Xs_batch = np.zeros([1,mm,4],dtype='int32')
        Xs_batch[0,0:mm,:] = movei
        T_batch = np.zeros([1])
        T_batch[0] = mm
        if T_batch > 0:
            action_p = self.sess.run(self.rollout_p, feed_dict = { self.x:Chess_batch, self.y_:Xs_batch, self.T:T_batch, self.cy:T_batch, self.keep_prob:1, self.is_training:False } )
            return movei, action_p
        else:
            return movei, 0


class value_nn(object):
    
    def __init__(self):
        batch_size=1
        self.x = tf.placeholder('float',[batch_size,10,9,15])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        
        with tf.variable_scope('QNET'):
            self.vl_p = nw.Model3(self.x, self.keep_prob, self.is_training)
        vlist=tf.global_variables()
        
        VNlist={}
        for i in range(len(vlist)):
            vname=vlist[i].name
            if ('QNET' in vname) and ('RMSProp' not in vname):
                VNlist[vname.replace('QNET/','').replace(':0','')]=vlist[i]

        saver=tf.train.Saver(VNlist)
        self.sess=tf.Session()
        saver.restore(self.sess,'./param/QNET.ckpt')
    
    def __call__(self, pos):
        Chess_batch = np.zeros([1,10,9,15])
        Chess_batch[0,:,:,0] = pos+7
        Chess_batch[0,:,:,1:8] = cs.feature_map(pos)
        Chess_batch[0,:,:,8:15] = -np.flipud(cs.feature_map(-np.flipud(pos)))
        
            
        value = self.sess.run(self.vl_p, feed_dict = { self.x:Chess_batch, self.keep_prob:1, self.is_training:False } )
        return value
