import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
def cnn_1d(input_tensor):
    input_tensor=tf.reshape(input_tensor,[-1,1,1024,1]) 
    with tf.variable_scope('layer01-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [1, 12, 1, 8], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))  
        conv1_biases = tf.get_variable("bias", [8], initializer=tf.constant_initializer(0.1))  
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases)) 
    with tf.name_scope("layer02-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,1,2,1],strides=[1,1,2,1],padding="SAME")
    with tf.variable_scope("layer03-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [1, 3, 8, 16],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))  
    with tf.name_scope("layer04-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        cnn_out = tf.reshape(pool2, [-1, nodes])
    return pool1,pool2,cnn_out

def cnn_2d(input_tensor):
    input_tensor=tf.reshape(input_tensor,[-1,32,32,1])
    with tf.variable_scope('layer11-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [3, 3, 1, 8],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [8], initializer=tf.constant_initializer(0.1))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.name_scope("layer12-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
    with tf.variable_scope("layer13-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [3, 3, 8, 16],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.name_scope("layer14-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        cnn_out = tf.reshape(pool2, [-1, nodes])
    return pool1,pool2,cnn_out  

def feature_extract_and_output(cnn_out,dropout_placeholdr):
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































