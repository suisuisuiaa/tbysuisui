from PIL import Image
import scipy.io as scio
from pylab import *
import os
path = r'E:\fault-diagnosis\0HP'
filenames = os.listdir(path)   #提取十个类型的mat文件文件名
for item in filenames:
    file_path = os.path.join(path, item)  #os.path.join是把文件名和文件的路径拼接到一起
    file = scio.loadmat(file_path)  #读取matlab文件
    for key in file.keys():
        if 'DE' in key:           #提取DE端数据
            X = file[key]
            for i in range(50):
                length = 1024
                all_lenght = len(X)  #DE端数据总长度
                random_start = np.random.randint(low=0, high=(all_lenght - 2*length))
                sample = X[random_start:random_start + length] 
                sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))  #归一化
                sample = np.round(sample*255.0)  #转换为灰度像素值
                sample = sample.reshape(32,32)
                im = Image.fromarray(sample)
                im.convert('L').save('E:\\CWRU\\灰度图\\'+str(key)+str(i)+'.jpg',format = 'jpeg')
