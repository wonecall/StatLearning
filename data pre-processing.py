from PIL import Image
import numpy as np
import os
import re
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
from sklearn.preprocessing import OneHotEncoder

def generate_train_dataset():
    dataet_dir = r'C:\Users\adam\Desktop\StaTest\train_data'
    class_list = os.listdir(dataet_dir)
    print(class_list)
    label_name_dict={}
    for label,class_name in enumerate(class_list):
        label_name_dict[label] = class_name
    inputs=[]
    labels=[]
    for label,class_name in enumerate(class_list):
        class_dir = os.path.join(dataet_dir,class_name)     
        image_list = os.listdir(class_dir)
        for img_name in image_list:
            img = Image.open(os.path.join(class_dir,img_name))
            img_array=np.array(img)
            inputs.append(img_array)
            labels.append(label)
        print(class_name,len(image_list))
    inputs=np.array(inputs)
    labels=np.array(labels)
    print(inputs.shape)
    print(labels.shape)
    np.save(r'C:\Users\adam\Desktop\StaTest\train_inputs.npy',inputs)
    np.save(r'C:\Users\adam\Desktop\StaTest\train_labels.npy',labels)

def generate_test_dataset():
    dataet_dir = r'C:\Users\adam\Desktop\StaTest\released_test'
    image_list=os.listdir(dataet_dir)
    inputs=[]
    for i in range(len(image_list)):
        image_name=str(i)+'.png'
        img = Image.open(os.path.join(dataet_dir,image_name))
        img_array=np.array(img)
        inputs.append(img_array)
    inputs=np.array(inputs)
    print(inputs.shape)
    np.save(r'C:\Users\adam\Desktop\StaTest\test_inputs.npy',inputs)

def data_augment(filename):

    image=imageio.imread(filename)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug) 
    seq=iaa.Sequential([
        iaa.Fliplr(0.5), # 左右反转
        iaa.Flipud(0.3), # 上下翻转
        sometimes(iaa.Affine(rotate=(-45, 45))),
        sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})), # 旋转仿射
        sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})) # 缩放仿射
    ])

    images_aug = [seq.augment_image(image) for _ in range(10)]
    return images_aug

def generate_augment_image():
    dataet_dir = r'C:\Users\adam\Desktop\StaTest\train_data'
    class_list = os.listdir(dataet_dir)
    inputs = []
    labels = []
    for label, class_name in enumerate(class_list):
        class_dir = os.path.join(dataet_dir, class_name)
        image_list = os.listdir(class_dir)
        for img_name in image_list:
            img_aug_list = data_augment(os.path.join(class_dir, img_name))
            inputs+=img_aug_list
            labels+=[label for i in range(10)]
    inputs = np.array(inputs)
    labels = np.array(labels)
    print(inputs.shape)
    print(labels.shape)
    np.save(r'C:\Users\adam\Desktop\StaTest\train_inputs_aug.npy', inputs)
    np.save(r'C:\Users\adam\Desktop\StaTest\train_labels_aug.npy', labels)
   
if __name__=="__main__":
    generate_train_dataset()
    generate_test_dataset()
    labels= np.load(r'C:\Users\adam\Desktop\StaTest\test_inputs.npy')
    generate_augment_image()

