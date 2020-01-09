import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd
from PIL import Image


inputs=np.load(r'C:\Users\adam\Desktop\StaTest\train_inputs.npy')
labels= np.load(r'C:\Users\adam\Desktop\StaTest\train_labels.npy')
inputs_aug=np.load(r'C:\Users\adam\Desktop\StaTest\train_inputs_aug.npy')
labels_aug= np.load(r'C:\Users\adam\Desktop\StaTest\train_labels_aug.npy')
inputs=np.concatenate([inputs,inputs_aug],axis=0)
labels=np.concatenate([labels,labels_aug],axis=0)

# one-hot
enc=OneHotEncoder()
labels=np.expand_dims(labels,-1)   #在-1轴（最后一轴）上扩维
enc.fit(labels)
labels=enc.transform(labels).toarray()
# normalization
inputs=inputs.astype("float32")/255
# 打乱数据
permutation = np.random.permutation(inputs.shape[0])  
inputs, labels = inputs[permutation], labels[permutation]
# 分训练集和测试集
train_data=inputs[:60000]
train_label=labels[:60000]
test_data=inputs[60000:]
test_label=labels[60000:]

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})   
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, 28, 28])        
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])           

W_conv1 = weight_variable([5, 5, 1, 32])              
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)           
h_pool1 = max_pool_2x2(h_conv1)                                   
W_conv2 = weight_variable([5,5, 32, 64])                # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)            # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                     # output size 7x7x64
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 15])
b_fc2 = bias_variable([15])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()   
logs_train_dir = r'C:\Users\adam\Desktop\StaTest\modelSaver'
with tf.Session() as sess:
    print('Reading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]      
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')

    test_data=np.load(r'C:\Users\adam\Desktop\StaTest\test_inputs.npy')
    split_size = 100
    batch_count = test_data.shape[0] // split_size
    batches_data = np.split(test_data,batch_count )         #把test_data数组从左到右按batch_count大小顺序切分
    print("Batch num:", batch_count)

    y_pres = []
    y_pres1 = []
    for i in range(batch_count):
        batch_test_data = test_data[split_size*i : split_size*(i+1)]
        y_pres = sess.run(prediction,feed_dict = {xs:batch_test_data, keep_prob: 1.})
        y_index = np.argmax(y_pres, axis = -1)   #返回最大值所在索引
        y_pres1.append(y_index)

    y_pres1 = (np.array(y_pres1)).flatten()  ##默认按行的方向降维
    id = np.arange(y_pres1.shape[0])
    data = np.stack([id, y_pres1],axis=1)   #沿着新轴axis=1连接数组序列
    df = pd.DataFrame(data, columns =["id","categories"])
    df.to_csv(r'C:\Users\adam\Desktop\StaTest\aaaaab.csv',index=False)
    