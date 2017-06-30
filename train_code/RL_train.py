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
        y=y+tf.concat(3,[x,tf.zeros([xsize[0],xsize[1],xsize[2],fout-fin])])
    else:
        with tf.variable_scope('res_proj'):
            w=tf.get_variable(name='res_proj_w',shape=[1,1,fin,fout],dtype='float')
            y=y+conv2d(x,w)
    return y
def Model(x,y_,cy,T,z,M,is_training):
    xsize=x.get_shape().as_list()
    batch_size=xsize[0]
    fin=xsize[-1]
    fout=64
    is_training=tf.constant(False)
    with tf.variable_scope('Policy'):
        tf.add_to_collection('ema_op',tf.no_op())
        with tf.variable_scope('pre_block'):
            with tf.variable_scope('pre1'):
                y=block(x,5,fin)
            with tf.variable_scope('pre_att'):
                yatt=block(y,5,fin)
            att1=tf.nn.softmax(yatt,dim=1)
            att2=tf.nn.softmax(yatt,dim=2)
            y=tf.concat(3,[y,y*att1,y*att2])
            #y=tf.nn.dropout(y,keep_prob)
            y,ema=bn(y,3*fin,is_training,True)
            tf.add_to_collection('ema_op',ema)
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
            y=tf.concat(3,[y,y*att1,y*att2])
        with tf.variable_scope('output'):
            #yin=tf.nn.dropout(y,keep_prob)
            with tf.variable_scope('block_1'):
                with tf.variable_scope('y1_0'):
                    y1=block(y,3,fout)
                with tf.variable_scope('y1'):
                    y1=block(y1,1,fout)
                #b1=tf.get_variable(name='bias_1',initializer=tf.zeros([90,1]))
            with tf.variable_scope('block_2'):
                with tf.variable_scope('y2_0'):
                    y2=block(y,3,fout)
                with tf.variable_scope('y2'):
                    y2=block(y2,1,fout)
                #b2=tf.get_variable(name='bias_2',initializer=tf.zeros([90,1]))
        w=tf.get_variable(name='pw',shape=[4*fout,1],dtype='float')
        batch_p=[]
        for i in range(batch_size):
            #print(i)
            y1id=y_[i,0:T[i],0]*9+y_[i,0:T[i],1]
            y2id=y_[i,0:T[i],2]*9+y_[i,0:T[i],3]
            y1_lookup=tf.nn.embedding_lookup(tf.reshape(y1[i,:,:,:],[90,-1]),y1id)
            y2_lookup=tf.nn.embedding_lookup(tf.reshape(y2[i,:,:,:],[90,-1]),y2id)
            p=tf.matmul(tf.concat(1,[y1_lookup,y2_lookup,y1_lookup-y2_lookup,y1_lookup*y2_lookup]),w)[:,0]
            p=tf.cond(T[i]>tf.constant(0),lambda:tf.nn.softmax(p),lambda:tf.constant(0.0))
            loss_value=tf.cond(T[i]>tf.constant(0),lambda:-tf.log(p[cy[i]]+1e-5)*z[i],lambda:tf.constant(0.0))
            tf.add_to_collection(name='losses',value=loss_value)
            batch_p.append(p)
        loss=tf.add_n(tf.get_collection('losses'),name='loss')/batch_size
    return batch_p,loss,tf.get_collection('ema_op')
def choose_step(p,plist):
    p0=0.0
    j=0
    while ((p0<=p) and (j<np.shape(plist)[0])):
        p0+=plist[j]
        j+=1
    return int(j-1)
if (__name__=='__main__'):
    batch_size=200
    step_per_batch=100
    x=tf.placeholder('float',[batch_size,10,9,29])
    y_=tf.placeholder(tf.int32,[batch_size,120,4])
    T=tf.placeholder(tf.int32,[batch_size])
    cy=tf.placeholder(tf.int32,[batch_size])
    z=tf.placeholder('float',[batch_size])
    #keep_prob=tf.placeholder(tf.float32)
    is_training=tf.placeholder(tf.bool)
    with tf.variable_scope('F'):
        batch_pF,loss0,ema0=Model(x,y_,cy,T,z,120,is_training)
    print('Player1 established')
    with tf.variable_scope('T'):
        batch_pT,loss,ema=Model(x,y_,cy,T,z,120,is_training)
    print('Player2 established')
    global_step=tf.Variable(0,trainable=False)
    lr=tf.train.exponential_decay(1e-3,global_step,1000,0.6,staircase=True)
    #ema.append(tf.assign_add(global_step,1))
    ema=[tf.assign_add(global_step,1)]
    #lr=1e-2
    #opt=tf.train.AdadeltaOptimizer(lr)
    #lr=0.005
    opt=tf.train.AdadeltaOptimizer(lr)
    gradient_all=opt.compute_gradients(loss)
    grads_vars=[v for (g,v) in gradient_all if g is not None]
    with tf.control_dependencies(ema):
        gradient=opt.compute_gradients(loss,grads_vars)
    grads_holder=[(tf.placeholder(tf.float32,shape=g.get_shape()),v) for (g,v) in gradient]
    train_op=opt.apply_gradients(grads_holder)
    Tvlist={}
    Fvlist={}
    vlist=tf.global_variables()
    for i in range(len(vlist)):
        vname=vlist[i].name
        if (vname[0]=='T') and ('Policy' in vname) and ('Adadelta' not in vname):
            Tvlist[vname.replace('T/','')]=vlist[i]
        if (vname[0]=='F') and ('Policy' in vname) and ('Adadelta' not in vname):
            Fvlist[vname.replace('F/','')]=vlist[i]
        #print(vname)
    Tsaver=tf.train.Saver(Tvlist)
    Fsaver=tf.train.Saver(Fvlist)
    tf_config=tf.ConfigProto(allow_soft_placement=True)  
    tf_config.gpu_options.per_process_gpu_memory_fraction=0.5
    sess=tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    Tsaver.restore(sess,'./rl00.ckpt')
    Fsaver.restore(sess,'./rl00.ckpt')
    op_version=0
    for i in range(5001):
        print('batch id:',i,'opponent version:',op_version)
        X_batch=np.int32(np.zeros([batch_size,10,9]))
        Xs_batch=np.int32(np.zeros([batch_size,120,4]))
        T_batch=np.int32(np.zeros([batch_size]))
        Chess_batch=np.zeros([batch_size,10,9,29])
        #new_chess_games
        result=np.zeros([batch_size])
        for j in range(batch_size):
            X_batch[j,:,:]=ChessStep.new_chess_game()
            Xi=X_batch[j,:,:]
            Chess_batch[j,:,:,0:15]=ChessStep.gen_x(Xi)
            Chess_batch[j,:,:,15:22]=ChessStep.feature_map(Xi)
            Chess_batch[j,:,:,22:29]=-np.flipud(ChessStep.feature_map(-np.flipud(Xi)))
            movei=ChessStep.gen_next_moves(X_batch[j,:,:])
            Xs_batch[j,0:np.shape(movei)[0],:]=movei
            T_batch[j]=np.shape(movei)[0]
        action_p=sess.run(batch_pF,feed_dict={x:Chess_batch,y_:Xs_batch,T:T_batch,is_training:False})
        for j in range(100,batch_size):
            if (T_batch[j]>0) and (result[j]==0):
                p=random.random()
                stepid=choose_step(p,action_p[j])
                stepmove=Xs_batch[j,stepid,:]
                X_batch[j,stepmove[2],stepmove[3]]=X_batch[j,stepmove[0],stepmove[1]]
                X_batch[j,stepmove[0],stepmove[1]]=0
                X_batch[j,:,:]=-np.flipud(X_batch[j,:,:])
        #play games
        Saved_X=np.int32(np.zeros([step_per_batch,batch_size,10,9]))
        Saved_Xstep=np.int32(np.zeros([step_per_batch,batch_size]))
        stepnum_per_batch=np.int32(np.zeros([batch_size]))
        for xq_step in range(step_per_batch):
            if (xq_step%10==0):
                print(xq_step,'win:',np.sum(result>0),'lose:',np.sum(result<0))
            Chess_batch=np.zeros([batch_size,10,9,29])
            Xs_batch=np.int32(np.zeros([batch_size,120,4]))
            T_batch=np.int32(np.zeros([batch_size]))
            for j in range(batch_size):
                if (result[j]==0):
                    Xi=X_batch[j,:,:]
                    Chess_batch[j,:,:,0:15]=ChessStep.gen_x(Xi)
                    Chess_batch[j,:,:,15:22]=ChessStep.feature_map(Xi)
                    Chess_batch[j,:,:,22:29]=-np.flipud(ChessStep.feature_map(-np.flipud(Xi)))
                    movei=ChessStep.gen_next_moves(X_batch[j,:,:])
                    Xs_batch[j,0:np.shape(movei)[0],:]=movei
                    T_batch[j]=np.shape(movei)[0]
            action_p=sess.run(batch_pT,feed_dict={x:Chess_batch,y_:Xs_batch,T:T_batch,is_training:False})
            #print(T_batch,action_p)
            for j in range(batch_size):
                Saved_X[xq_step,j,:,:]=np.copy(X_batch[j,:,:])
                if (T_batch[j]==0) and (result[j]==0):
                    result[j]=-1
                    stepnum_per_batch[j]=xq_step
                if (T_batch[j]>0) and (result[j]==0):
                    p=random.random()
                    stepid=choose_step(p,action_p[j])
                    #print(stepid)
                    Saved_Xstep[xq_step,j]=stepid
                    stepmove=Xs_batch[j,stepid,:]
                    X_batch[j,stepmove[2],stepmove[3]]=X_batch[j,stepmove[0],stepmove[1]]
                    X_batch[j,stepmove[0],stepmove[1]]=0
                    if (-7 not in X_batch[j,:,:]):
                        result[j]=1
                    else:
                        X_batch[j,:,:]=-np.flipud(X_batch[j,:,:])
            Chess_batch=np.zeros([batch_size,10,9,29])
            Xs_batch=np.int32(np.zeros([batch_size,120,4]))
            T_batch=np.int32(np.zeros([batch_size]))
            for j in range(batch_size):
                if (result[j]==0):
                    Xi=X_batch[j,:,:]
                    Chess_batch[j,:,:,0:15]=ChessStep.gen_x(Xi)
                    Chess_batch[j,:,:,15:22]=ChessStep.feature_map(Xi)
                    Chess_batch[j,:,:,22:29]=-np.flipud(ChessStep.feature_map(-np.flipud(Xi)))
                    movei=ChessStep.gen_next_moves(X_batch[j,:,:])
                    Xs_batch[j,0:np.shape(movei)[0],:]=movei
                    T_batch[j]=np.shape(movei)[0]
            action_p=sess.run(batch_pF,feed_dict={x:Chess_batch,y_:Xs_batch,T:T_batch,is_training:False})
            for j in range(batch_size):
                if (T_batch[j]==0) and (result[j]==0):
                    result[j]=1
                    stepnum_per_batch[j]=xq_step+1
                if (T_batch[j]>0) and (result[j]==0):
                    p=random.random()
                    stepid=choose_step(p,action_p[j])
                    stepmove=Xs_batch[j,stepid,:]
                    X_batch[j,stepmove[2],stepmove[3]]=X_batch[j,stepmove[0],stepmove[1]]
                    X_batch[j,stepmove[0],stepmove[1]]=0
                    if (-7 not in X_batch[j,:,:]):
                        result[j]=-1
                    X_batch[j,:,:]=-np.flipud(X_batch[j,:,:])
        #redo
        print('Finished','win:',np.sum(result>0),'lose:',np.sum(result<0))
        result[result==0]=-0.1
        grad_per_step=[]
        for xq_step in range(step_per_batch):
            Chess_batch=np.zeros([batch_size,10,9,29])
            Xs_batch=np.int32(np.zeros([batch_size,120,4]))
            T_batch=np.int32(np.zeros([batch_size]))
            for j in range(batch_size):
                if (xq_step<stepnum_per_batch[j]):
                    Xi=Saved_X[xq_step,j,:,:]
                    Chess_batch[j,:,:,0:15]=ChessStep.gen_x(Xi)
                    Chess_batch[j,:,:,15:22]=ChessStep.feature_map(Xi)
                    Chess_batch[j,:,:,22:29]=-np.flipud(ChessStep.feature_map(-np.flipud(Xi)))
                    movei=ChessStep.gen_next_moves(Xi)
                    Xs_batch[j,0:np.shape(movei)[0],:]=movei
                    T_batch[j]=np.shape(movei)[0]
                #print(T_batch[j])
            grad_per_step.append(sess.run(gradient,feed_dict={x:Chess_batch,y_:Xs_batch,T:T_batch,cy:Saved_Xstep[xq_step,:],z:result,is_training:True}))
        print('compute_grads finished')
        for xq_step in range(step_per_batch):
            grads_dict={}
            for j in range(len(grad_per_step[xq_step])):
                grads_dict[grads_holder[j][0]]=grad_per_step[xq_step][j][0]
            sess.run(train_op,feed_dict=grads_dict)
        print('update finished')
        #sess.run(step_op)
        if (i%5==0):
            Tsaver.save(sess,'rl'+str(int(i/20)+1)+'.ckpt')
            print('version '+str(int(i/20)+1)+' has been saved/updated.')
            op_version=random.randint(0,int(i/20))
            if (op_version>0):
                Fsaver.restore(sess,'./rl'+str(op_version)+'.ckpt')
            else:
                Fsaver.restore(sess,'./rl00.ckpt')
            np.savez('chess.npz',Saved_X)
        '''
        for xq_step in range(60):
            for j in range(batch_size):
        if (i%10==0):
            X_batch,Xs_batch,T_batch,cy_batch=X.mini_batch(batch_size,is_tr=True,is_shuffle=True)
            print(i,sess.run([loss,acr],feed_dict={x:X_batch,y_:Xs_batch,T:T_batch,cy:cy_batch,is_training:False}))
        if (i%1000==0):
            ac_sum=0.0
            batch_num=int(1723/batch_size)
            for j in range(batch_num):
                X_batch,Xs_batch,T_batch,cy_batch=X.mini_batch(batch_size,is_tr=False,is_shuffle=False,st=j*batch_size)
                ac_sum+=sess.run(acr,feed_dict={x:X_batch,y_:Xs_batch,T:T_batch,cy:cy_batch,is_training:False})
            print(i,ac_sum/(batch_size*batch_num))
        '''
sess.close()