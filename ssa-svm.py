# -*- coding: utf-8 -*-
import PIL.Image
import numpy as np    
import matplotlib.pyplot as plt 
from sklearn import svm 
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import SSA as SSA
from sklearn.metrics import accuracy_score

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')  
    plt.title(title) 
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name)   
    plt.yticks(num_local, labels_name)   
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.savefig('cnn_image/confusion_matrinx_svm.jpg')
    plt.show()

def fun(X):
    # 训练svm分类器
    clf = svm.SVC(C=X[0], kernel='rbf', gamma=X[1])  # ovr:一对多策略
    clf.fit(train_X, train_Y) 
    # 计算svc分类器的准确率
    tes_label = clf.predict(test_X) 
    train_labelout = clf.predict(train_X)  
    val_label = clf.predict(valid_X)  
    output = 3 - accuracy_score(test_Y, tes_label) - accuracy_score(train_Y,train_labelout) - accuracy_score(valid_Y,val_label)  # 计算错误率，如果错误率越小，结果越优
    return output

# 加载数据
mode='2d' # 选择 1d 2d 还是 fusion 对应尽1dcnn 2dcnn 与融合cnn
dataFile = 'result/'+mode+'data_feature.mat'
data = loadmat(dataFile)
train_X=data['train_X']
train_Y=data['train_Y'].argmax(axis=1)
valid_X=data['valid_X']
valid_Y=data['valid_Y'].argmax(axis=1)
test_X=data['test_X']
test_Y=data['test_Y'].argmax(axis=1)


ss=StandardScaler().fit(train_X)
train_X=ss.transform(train_X)
valid_X=ss.transform(valid_X)
test_X=ss.transform(test_X)

#设置麻雀参数
pop = 20 #种群数量
MaxIter = 50 #最大迭代次数
dim = 2 #维度
lb = np.matrix([[0.1],[0.1]]) #下边界
ub = np.matrix([[200],[200]])#上边界
fobj = fun
GbestScore,GbestPositon,Curve = SSA.SSA(pop,dim,lb,ub,MaxIter,fobj)
print('最优适应度值：',GbestScore)
print('c,g最优解：',GbestPositon)

clf=svm.SVC(C=GbestPositon[0,0],kernel='rbf',gamma=GbestPositon[0,1]) # ovr:一对多策略
clf.fit(train_X,train_Y)
tra_label=clf.predict(train_X) 
tes_label=clf.predict(test_X) 
val_label=clf.predict(valid_X)
print("训练集准确率：", accuracy_score(train_Y,tra_label) )
print("测试集准确率：", accuracy_score(test_Y,tes_label) )
print("验证集准确率：", accuracy_score(valid_Y,val_label) )

print('混淆矩阵')
cm=confusion_matrix(test_Y,tes_label)
print(cm)
for i in range(10):
    precise=cm[i,i]/sum(cm[:,i])
    recall=cm[i,i]/sum(cm[i,:])
    f1=2*precise*recall/(precise+recall)
    print('测试集中，第',i,
          '类样本的精确率：',precise,
          ' 召回率为：',recall,
          ' F1分数为：',f1)

#混淆矩阵
labels_name=['0','1','2','3','4','5','6','7','8','9']
plot_confusion_matrix(cm, labels_name, "Classification confusion matrix")

plt.figure(1)
plt.plot(test_Y,'*',label = "True")
plt.plot(tes_label,'o',label = "predict")
plt.xlabel("Test Case")
plt.ylabel("Case Label")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.savefig('cnn_image/' + mode + 'cnn-ssa-svm.jpg')


#绘制适应度曲线
plt.figure(2)
plt.plot(Curve,'r-',linewidth=2)
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid()
plt.title('SSA',fontsize='large')
plt.show()
