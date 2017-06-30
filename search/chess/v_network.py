#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YongfengLi
"""

from . import ChessStep2 as cs    
from . import valuenet as nw   
import tensorflow as tf
import numpy as np


class value_nn(object):
    
    def __init__(self):
        batch_size=1
        self.x = tf.placeholder('float',[batch_size,10,9,15])
        #self.y =tf.placeholder(tf.float32,[batch_size])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        
        with tf.variable_scope('QNET'):
            self.vl_p = nw.Model2(self.x, self.keep_prob, self.is_training)
        vlist=tf.global_variables()
        
        VNlist={}
        for i in range(len(vlist)):
            vname=vlist[i].name
            if ('QNET' in vname) and ('RMSProp' not in vname):
                VNlist[vname.replace('QNET/','').replace(':0','')]=vlist[i]

        saver3=tf.train.Saver(VNlist)
        self.sess=tf.Session()
        saver3.restore(self.sess,'./param/QNET.ckpt')
    
    def __call__(self, pos):
        Chess_batch = np.zeros([1,10,9,15])
        Chess_batch[0,:,:,0] = pos+7
        Chess_batch[0,:,:,1:8] = cs.feature_map(pos)
        Chess_batch[0,:,:,8:15] = -np.flipud(cs.feature_map(-np.flipud(pos)))
        Xs_batch = np.zeros([1],dtype='float32') 
            
        value = self.sess.run(self.vl_p, feed_dict = { self.x:Chess_batch, self.keep_prob:1, self.is_training:False } )
        return value
