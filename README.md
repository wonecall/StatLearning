# Introduction
This is our Tensorflow implementation for the paper:Predicting the category for Quick Draw image using statistical learning algorithms.   
Author: Ke Wang (onecall at sjtu.edu.cn)  


# Environment Requirement
The code has been tested running under Python 3.6.9. The required packages are as follows:  
tensorflow == 1.14.0  
keras == 2.3.1  
sklearn == 0.22  
bottleneck == 1.3.1  
numpy == 1.17.4  

# Run the Codes
The data pre-processing run the file data pre-processing.py.We adopt Imgaug data enhancement library to augment our dataset. Imgaug is an encapsulated python library for image augmentation that supports keypoint and bounding box transformations. By using this method, we increase the images to 60000.    
    python data pre-processing.py  
    
The KNN,SVM and Random Forest model run the file KNN&SVM&Random Forest.py.   
    python KNN&SVM&Random Forest.py  
    
The CNN model training run the file CNN-Model Train.py. It can save the best model for test.  
    python CNN-Model Train.py  
    
The CNN model test run the file CNN-Model Test.py. It can test the saved model for accurary.  
    python CNN-Model Test.py  
    
The InceptionV3 model training run the file InceptionV3-Model Train.py. It can save the best model for test.   
    python InceptionV3-Model Train.py  
    
The InceptionV3 model test run the file InceptionV3-Model Test.py. It can test the saved model for accurary.  
    python InceptionV3-Model Test.py  
    
# Result 

Method  | Accuracy  
---- | ----- 
KNN  | 0.637 
SVM  | 0.682
RF  | 0.663 
CNN | 0.783
Inception V3 | 0.822

