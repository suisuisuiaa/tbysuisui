# -*- coding: utf-8 -*-
from scipy.io import loadmat,savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from model import CNN_1D,CNN_2D,CNN_fusion
from sklearn.preprocessing import StandardScaler 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' 

import tensorflow as tf
tf.set_random_seed(0)

data=loadmat('result/data_process.mat')
train_X=data['train_X']
train_Y=data['train_Y']
valid_X=data['valid_X']
valid_Y=data['valid_Y']
test_X=data['test_X']
test_Y=data['test_Y']

ss=StandardScaler().fit(train_X)
train_X=ss.transform(train_X)
valid_X=ss.transform(valid_X)
test_X=ss.transform(test_X)

x = tf.placeholder(tf.float32, [None, 1024])
y = tf.placeholder(tf.float32, [None,10])
dropout_placeholdr = tf.placeholder(tf.float32)
# In[2] 定义网络相关参数
# 定义参数
learning_rate = 0.001#初始学习率
num_epochs=100#迭代次数
batch_size = 64#batch_size
# 选择 1d 2d 还是 fusion 对应尽1dcnn 2dcnn 与融合cnn
mode='2d' 
if mode=='1d':
    _,_,feature,pred= CNN_1D(x,dropout_placeholdr)
elif mode=='2d':
    _,_,feature,pred= CNN_2D(x,dropout_placeholdr)
elif mode=='fusion':
    _,_,_,_,_,feature,pred= CNN_fusion(x,dropout_placeholdr)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=pred)
cost = tf.reduce_mean(cross_entropy)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) # tf.argmax()
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-10).minimize(cost) 

saver = tf.train.Saver()
train = []
test = []
valid = []
trainacc=[]
testacc = []
validacc=[]
select=0 #为0则重新训练模型 1则加载模型直接进行softmax分类

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if select==0:#@重新训练模型
        print("训练模式")#

        n_samples= train_X.shape[0] 
        batches=int(np.ceil(n_samples/batch_size))
        for epoch in range(num_epochs):
            rand_index=np.arange(n_samples)
            np.random.shuffle(rand_index) 
            for i in range(batches):
                index = rand_index[i*batch_size:(i+1)*batch_size]
                batch_x = train_X[index,:]
                batch_y = train_Y[index,:]
                sess.run(optimizer,feed_dict={x: batch_x, y: batch_y,dropout_placeholdr:0.5})
    
            
            [train_loss,train_acc] = sess.run([cost,accuracy], feed_dict={x: train_X, y: train_Y,dropout_placeholdr:1.0}) 
            [test_loss,test_acc] = sess.run([cost, accuracy],feed_dict={x: test_X, y: test_Y, dropout_placeholdr: 1.0})
            [valid_loss,valid_acc] = sess.run([cost,accuracy], feed_dict={x: valid_X, y: valid_Y,dropout_placeholdr:1.0})

    
            train.append(train_loss)
            test.append(test_loss)
            valid.append(valid_loss)

            trainacc.append(train_acc)
            testacc.append(test_acc)
            validacc.append(valid_acc)
            
        saver.save(sess, 'save_model_cnn/'+mode)
        plt.figure()
        plt.plot( train, c='r',label='train')
        plt.plot(test, c='b', label='test')
        plt.plot( valid, c='g',label='valid')
        plt.ylabel('cross entropy')
        plt.xlabel('Epoch')
        plt.legend() 
        plt.title('loss curve')
        plt.savefig('cnn_image/'+mode+'loss curve.jpg')
            
        plt.figure()
        plt.plot( trainacc, c='r',label='train')
        plt.plot(testacc, c='b', label='test')
        plt.plot( validacc, c='g',label='valid')
        plt.ylabel('accuracy')
        plt.xlabel('Epoch')
        plt.title('accuracy curve')
        plt.savefig('cnn_image/'+mode+'accuracy curve.jpg')
        plt.legend()
        plt.show()  

    
    elif select==1:#调用训练好的模型进行分类
        print("测试模式")#
        saver.restore(sess, 'save_model_cnn/'+mode) 
        train_acc = sess.run(accuracy, feed_dict={x: train_X, y: train_Y,dropout_placeholdr:1.0})
        valid_acc = sess.run(accuracy, feed_dict={x: valid_X, y: valid_Y,dropout_placeholdr:1.0})
        test_acc = sess.run(accuracy, feed_dict={x: test_X, y: test_Y,dropout_placeholdr:1.0})
        print('训练集分类精度为：',train_acc*100,'%')
        print('验证集分类精度为：',valid_acc*100,'%')
        print('测试集分类精度为：',test_acc*100,'%')

    else:#调用训练好的模型进行特征提取
        print("特征提取模式")
        saver.restore(sess, 'save_model_cnn/'+mode)
        train_feature = sess.run(feature, feed_dict={x: train_X,dropout_placeholdr:1.0})
        valid_feature = sess.run(feature, feed_dict={x: valid_X,dropout_placeholdr:1.0})
        test_feature = sess.run(feature, feed_dict={x: test_X,dropout_placeholdr:1.0})
        

        
        savemat('result/'+mode+'data_feature.mat', {'train_X': train_feature,'train_Y': train_Y,
                               'valid_X': valid_feature,'valid_Y': valid_Y,
                               'test_X': test_feature,'test_Y': test_Y}) 
 
    
    
    
    
    
    
