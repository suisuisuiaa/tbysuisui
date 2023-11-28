# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from model import CNN_1D, CNN_2D,CNN_fusion
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from sklearn.manifold import TSNE  # t-SNE is a tool to visualize high-dimensional data
from mpl_toolkits.mplot3d import Axes3D


import tensorflow as tf
tf.set_random_seed(0) 


# In[] 加载数据

data=loadmat('result/data_process.mat')
train_X=data['train_X']
train_Y=data['train_Y'].argmax(axis=1) # returns the indices of the maximum values along an axis
valid_X=data['valid_X']
valid_Y=data['valid_Y'].argmax(axis=1)
test_X=data['test_X']
test_Y=data['test_Y'].argmax(axis=1)

ss=StandardScaler().fit(train_X) #  perform standardization by centering and scaling ,compute the mean and std to be used for later scaling
train_X=ss.transform(train_X)  #  fit + transform = fit_transform
valid_X=ss.transform(valid_X)
test_X=ss.transform(test_X)


input_data=test_X
label=test_Y

#原始数据可视化
method=TSNE(n_components=3)
feature0=method.fit_transform(input_data) 
colors = ['black', 'blue', 'purple', 'yellow', 'cadetblue', 'red', 'lime', 'cyan', 'orange', 'gray']
plt.figure()
ax = plt.axes(projection='3d') 
for i in range(len(colors)):
    ax.scatter3D(feature0[:, 0][label==i], feature0[:, 1][label==i],feature0[:, 2][label==i], c=colors[i],label=str(i))
    #ax.text(np.mean(feature0[:, 0][label==i]), np.mean(feature0[:, 1][label==i]),np.mean(feature0[:, 2][label==i]), str(i))
plt.legend(loc=2,fontsize="medium")
plt.title('original data')
plt.savefig('cnn_image/'+'original data可视化.jpg')


x = tf.placeholder(tf.float32, [None, 1024])
y = tf.placeholder(tf.float32, [None,10])
dropout_placeholdr = tf.placeholder(tf.float32)

mode='fusion' # 选择 1d 2d 还是 fusion 对应尽1dcnn 2dcnn 与融合cnn
if mode=='1d':
    cnn1d_pool1,cnn1d_pool2,feature,_= CNN_1D(x,dropout_placeholdr)
elif mode=='2d':
    cnn2d_pool1,cnn2d_pool2,feature,_= CNN_2D(x,dropout_placeholdr)
elif mode=='fusion':
    cnn1d_pool1,cnn1d_pool2,cnn2d_pool1,cnn2d_pool2,cnn_fusion,feature,_= CNN_fusion(x,dropout_placeholdr)

saver = tf.train.Saver() 


with tf.Session() as sess:
    saver.restore(sess, 'save_model_cnn/'+mode)
    name='cnn_fusion'
    if name=='cnn1d_pool1':
        features = sess.run(cnn1d_pool1, feed_dict={x: input_data,dropout_placeholdr:1.0})
    elif name=='cnn1d_pool2':
        features = sess.run(cnn1d_pool2, feed_dict={x: input_data,dropout_placeholdr:1.0})
    elif name=='cnn2d_pool1':
        features = sess.run(cnn2d_pool1, feed_dict={x: input_data,dropout_placeholdr:1.0})
    elif name=='cnn2d_pool2':
        features = sess.run(cnn2d_pool2, feed_dict={x: input_data,dropout_placeholdr:1.0})
    elif name=='cnn_fusion':
        features = sess.run(cnn_fusion, feed_dict={x: input_data,dropout_placeholdr:1.0})
    elif name=='feature':
        features = sess.run(feature, feed_dict={x: input_data,dropout_placeholdr:1.0})
    
    output_data=StandardScaler().fit_transform(features.reshape(input_data.shape[0],-1))
    feature1=method.fit_transform(output_data) 

    plt.figure()
    ax = plt.axes(projection='3d') 
    for i in range(len(colors)):
        ax.scatter3D(feature1[:, 0][label==i], feature1[:, 1][label==i],feature1[:, 2][label==i], c=colors[i],label=str(i))
        
    plt.legend(loc=2,fontsize="small")
    plt.title(name)
    plt.savefig('cnn_image/' + name + '可视化.jpg')
    plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
