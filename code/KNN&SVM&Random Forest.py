import numpy as np
import pandas as pd
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#K.set_image_dim_ordering('th')
K.image_data_format() == 'channels_first'

inputs=np.load(r'C:\Users\adam\Desktop\StaTest\train_inputs.npy')
labels= np.load(r'C:\Users\adam\Desktop\StaTest\train_labels.npy')
inputs=inputs.astype("float32")/255
permutation = np.random.permutation(inputs.shape[0])  
inputs, labels = inputs[permutation], labels[permutation]
x_image = np.reshape(inputs, [-1, 784])   
print(x_image.shape)
print(labels.shape)
X_train=x_image[:5400]
y_train=labels[:5400]
X_test=x_image[5400:]
y_test=labels[5400:]


#KNN
clf_knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
clf_knn.fit(X_train, y_train)
print(clf_knn)
y_pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print ('KNN accuracy: ',acc_knn)

#SGDClassifier
clf_sgd = SGDClassifier(loss='hinge', random_state=0) # loss='hinge' results in a linear SVM
clf_sgd.fit(X_train, y_train)
print(clf_sgd)
y_pred_sgd = clf_sgd.predict(X_test)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
print ('SVM stochastic gradient descent accuracy: ',acc_sgd)

#LinearSVC
clf_svm = LinearSVC(random_state=0)
clf_svm.fit(X_train, y_train)
print(clf_svm)
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print ('Linear SVM accuracy: ',acc_svm)

#Non-linear SVC
clf_svm = SVC(kernel='rbf', random_state=0) # using the Gaussian radial basis function
clf_svm.fit(X_train, y_train)
print(clf_svm)
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print ('RBF SVM accuracy: ',acc_svm)

#Random Forest
clf_rf = RandomForestClassifier(n_estimators=160, n_jobs=-1, random_state=0)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print ('Random forest accuracy: ',acc_rf)






