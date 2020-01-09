import tensorflow as tf
import pandas as pd
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

y_pres = []
labelList = []
lines = tf.gfile.FastGFile(r'E:\MyTensorflow\retrain\output_labels.txt').readlines()
uid_to_human = {}
for uid,line in enumerate(lines):
    line = line.strip('\n')
    uid_to_human[uid] = line
    
def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]

# 创建一个图来存放训练好的模型
with tf.gfile.FastGFile(r'E:\MyTensorflow\retrain\output_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # 拿到softmax的op
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    for root,dirs,files in os.walk(r'C:\Users\adam\Desktop\StaTest\released_test'):
        lenfiles= len(files)
        for i in range(lenfiles):
            filename = files[i]  # 获取文件名字符串
            filename1 = int(filename.split('.')[0])  # 以 . 分割提取文件名
            labelList.append(filename1)

        for file in files:
            print(file)
            image_data = tf.gfile.FastGFile(os.path.join(root,file),'rb').read()
            # 运行softmax节点，向其中feed值
            # 可以在网络中找到这个名字，DecodeJpeg/contents，
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
            predictions = np.squeeze(predictions)# 把结果转化为1维数据

            image_path = os.path.join(root, file)
            print(image_path)
            img = Image.open(image_path)
            top_k = predictions.argsort()[-5:][::-1]
            y_index = top_k[0]
            y_pres.append(y_index)

    print(y_pres)
    y_pres1 = (np.array(y_pres)).flatten()  ##默认按行的方向降维
    labelList = np.array(labelList)
    print(labelList)
    data = np.stack([labelList, y_pres1],axis=1)   #沿着新轴axis=1连接数组序列
    df = pd.DataFrame(data, columns =["id","categories"])
    ##保存csv格式数据
    df.to_csv(r'E:\MyTensorflow\retrain\resnet002.csv',index=False)
    print("ok")