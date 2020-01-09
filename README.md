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
python data pre-processing.py
The instruction of commands has been clearly stated in the codes (see the parser function in Model/utility/parser.py).

python KNN&SVM&Random Forest.py
python CNN-Model Train.py
python CNN-Model Test.py
python InceptionV3-Model Train.py
python InceptionV3-Model Test.py

