import tensorflow as tf
#import tensorflow.compat.v1 as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
#tf.disable_v2_behavior()
# In[] 建立网络,取全连接层的隐含层输出作为特征
def cnn_1d(input_tensor):
    input_tensor=tf.reshape(input_tensor,[-1,1,1024,1])  # 将tensor变换为参数shape的形式  tf.reshape(tensor, shape, name=None)
    # 第一层卷积层
    with tf.variable_scope('layer01-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [1, 12, 1, 8], #  shape是四维矩阵，前两个是卷积核尺寸（长宽），第三个是当前层的深度，第四个是过滤器的深度
            initializer=tf.truncated_normal_initializer(stddev=0.1))  # 生成截断正态分布的随机数
        conv1_biases = tf.get_variable("bias", [8], initializer=tf.constant_initializer(0.1))  # 偏置项初始化 偏置项的维度是下一层的深度，即过滤器的深度
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        #  conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None)
        # tf.nn.conv2d实现卷积前向传播
        #  padding:填充方式，VALID不添加，SAME为全0填充
        # input:当前层的节点矩阵——四维矩阵（batch_size,三维节点矩阵）；strides：不同维度上的步长
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))  #将输入小于0的值赋值为0，输入大于0的值不变
    # 第一层最大池化层
    with tf.name_scope("layer02-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,1,2,1],strides=[1,1,2,1],padding="SAME")
    # 第2层卷积层
    with tf.variable_scope("layer03-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [1, 3, 8, 16],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    # 第2层最大池化层    
    with tf.name_scope("layer04-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        # 全连接神经网络的输入是向量，而第二层池化层的输出是矩阵，将矩阵中的节点拉直转化为向量。其中通过get_shape函数可以得到矩阵的维度，返回的是元组
        # as_list()转化为列表
        # 只有张量才可以使用get_shape这种方法，tf.shape()都可以
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        cnn_out = tf.reshape(pool2, [-1, nodes])
    return pool1,pool2,cnn_out

def cnn_2d(input_tensor):
    # 原始震动数据每个样本维度为1024 正好可以转换为32*32的矩阵作为2d-cnn的输入
    input_tensor=tf.reshape(input_tensor,[-1,32,32,1])
    # 第一层卷积层
    with tf.variable_scope('layer11-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [3, 3, 1, 8],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [8], initializer=tf.constant_initializer(0.1))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # 第一层最大池化层
    with tf.name_scope("layer12-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
    # 第2层卷积层
    with tf.variable_scope("layer13-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [3, 3, 8, 16],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    # 第2层最大池化层    
    with tf.name_scope("layer14-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        cnn_out = tf.reshape(pool2, [-1, nodes])
    return pool1,pool2,cnn_out  

def feature_extract_and_output(cnn_out,dropout_placeholdr):
    # 最后是两层全连接层,特征提取就是提取的第一个全连接层的输出
    # 第二个全连接层是输出层 也就是分类层
    with tf.variable_scope('layer5-fc1'):
        weights1 = tf.get_variable('weights', [cnn_out.shape[1], 128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases1 = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.1))
        fc1=tf.matmul(cnn_out, weights1) + biases1
        fc1_drop = tf.nn.dropout(fc1, dropout_placeholdr)# dropput
        
    with tf.variable_scope('layer6-fc2'):
        weights2 = tf.get_variable('weights', [128, 10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases2 = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.1))
        output=tf.matmul(fc1_drop, weights2) + biases2
    return fc1,output


def CNN_1D(input_tensor,dropout_placeholdr):#仅1d-cnn
    pool1,pool2,cnn_out=cnn_1d(input_tensor)
    feature,output=feature_extract_and_output(cnn_out,dropout_placeholdr)
    return pool1,pool2,feature,output
def CNN_2D(input_tensor,dropout_placeholdr):#仅2d-cnn
    pool1,pool2,cnn_out=cnn_2d(input_tensor)
    feature,output=feature_extract_and_output(cnn_out,dropout_placeholdr)
    return pool1,pool2,feature,output

def CNN_fusion(input_tensor,dropout_placeholdr):#1d-cnn与2d融合
    cnn1d_pool1,cnn1d_pool2,cnn1d_cnn_out=cnn_1d(input_tensor)
    cnn2d_pool1,cnn2d_pool2,cnn2d_cnn_out=cnn_2d(input_tensor)
    
    cnn_fusion=tf.concat([cnn1d_cnn_out,cnn2d_cnn_out],axis=1)#融合cnn
    feature,output=feature_extract_and_output(cnn_fusion,dropout_placeholdr)
    return cnn1d_pool1,cnn1d_pool2,cnn2d_pool1,cnn2d_pool2,cnn_fusion,feature,output































