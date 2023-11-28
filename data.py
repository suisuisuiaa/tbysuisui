#coding:utf-8
from scipy.io import loadmat,savemat  
import numpy as np
import os
from sklearn import preprocessing  
from sklearn.preprocessing import MinMaxScaler

def capture(original_path):
    """读取mat文件，返回字典
    :param original_path: 读取路径
    :return: 数据字典
    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path) 
    files = {}
    n=0
    for i in filenames:
        # 文件路径
        file_path = os.path.join(d_path, i) 
        print(file_path,'为第',n,'类')
        n += 1
        file = loadmat(file_path) 
        file_keys = file.keys()
        for key in file_keys:
            if 'DE' in key:
                files[i] = file[key].ravel() # ravel() 
    return files

def slice_enc(data,number=1000,length=1024):
    """
    每个样本的长度为length 每类各number个样本
    """
    keys = data.keys() 
    labels,samples=[],[]
    n=0
    for i in keys:
        slice_data = data[i]
        for j in range(number):
            random_start = np.random.randint(low=0, high=(len(slice_data) - length)) 
            sample = slice_data[random_start:random_start + length]
            labels.append(n)
            samples.append(sample) 
        n += 1
    samples,lables=np.array(samples),np.array(labels)
    return samples,lables

# one-hot编码
def one_hot(labels):
    labels = np.array(labels).reshape([-1, 1])
    Encoder = preprocessing.OneHotEncoder()
    Encoder.fit(labels)
    labels = Encoder.transform(labels).toarray()
    labels = np.asarray(labels, dtype=np.int32)
    return labels

# 按rate中的比例划分训练集 验证集与测试集,比例相加为1
def train_valid_test_slice(data, labels,rate):
    
    nsamples=data.shape[0]
    index=np.arange(nsamples) # 数组范围创建数组
    np.random.shuffle(index) # 打乱数组顺序
    m1=int(nsamples*rate[0])
    m2=int(nsamples*(rate[0]+rate[1]))
    train_X=data[index[:m1],:]
    train_Y=labels[index[:m1],:]
    
    valid_X=data[index[m1:m2],:]
    valid_Y=labels[index[m1:m2],:]
    
    test_X=data[index[m2:],:]
    test_Y=labels[index[m2:],:]    

    return train_X,train_Y,valid_X, valid_Y, test_X, test_Y
if __name__ == "__main__":
    d_path='0HP/'
    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 将数据切分为需要的样本
    data, labels = slice_enc(data,number=1000,length=1024)
    # 类别转为onehot编码
    labels = one_hot(labels)
    
    # 将数据划分训练，验证集，测试集
    train_X,train_Y,valid_X, valid_Y, test_X, test_Y = train_valid_test_slice(data, labels,rate=[0.7, 0.2, 0.1])
    
    savemat("result/data_process.mat", {'train_X': train_X,'train_Y': train_Y,
                                   'valid_X': valid_X,'valid_Y': valid_Y,
                                   'test_X': test_X,'test_Y': test_Y}) 

    










