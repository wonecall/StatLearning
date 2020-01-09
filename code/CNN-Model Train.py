import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import time
import pandas as pd

# load dataset
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
#打乱数据
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

# 产生权重变量，符合 normal 分布
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义convolutional 图层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 定义 pooling 图层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 28, 28])       
ys = tf.placeholder(tf.float32, [None, 15])

keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])           # 最后一个1表示数据是黑白的

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32])                
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)           
# pooling
h_pool1 = max_pool_2x2(h_conv1)                                    

##conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64])               
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)        
h_pool2 = max_pool_2x2(h_conv2)                                  

##FC1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##FC2 layer ##
W_fc2 = weight_variable([1024, 15])
b_fc2 = bias_variable([15])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

prediction_clip = tf.clip_by_value(prediction,1e-8,1)
# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction_clip), reduction_indices=[1]))       
# optimizer
optimizer=tf.train.AdamOptimizer(1e-4)
tvars = tf.trainable_variables()
grads = tf.gradients(cross_entropy, tvars)  
max_grad_norm = 5
grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)  
global_step = tf.train.get_or_create_global_step()  
train_step = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step,
                                     name='train_op')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
epoch = 80
batch_size = 64

for i in range(epoch):
    permutation = np.random.permutation(train_data.shape[0]) 
    new_train_data, new_train_label = train_data[permutation], train_label[permutation]
    batch_num = new_train_data.shape[0]//batch_size
    for j in range(batch_num):
        batch_train_data = new_train_data[batch_size*j:batch_size*(j+1)]
        batch_train_label = new_train_label[batch_size*j:batch_size*(j+1)]
        sess.run(train_step, feed_dict={xs: batch_train_data, ys: batch_train_label, keep_prob: 0.7})
    print("epoch {},test_accuravcy:{:.4f}".format(i,compute_accuracy(test_data,test_label)))
    if i == 79:
        saver.save(sess,r'C:\Users\adam\Desktop\StaTest\modelSaver\model.ckpt',global_step=i)
        print('Modele保存成功')
        break
print("ok")
