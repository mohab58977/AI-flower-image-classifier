Flower-image-classifier
This project is part of the AI Programming with Python Nanodegree from [https://www.udacity.com/] to create and train a classifier using transfer learning from three different pre trained CNN to predict flowers images using a dataset with 102 classes.

The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on the dataset
Use the trained classifier to predict flower images
Dataset
A 102 category dataset, consisting of 102 flower categories, flowers commonly occuring in the United Kingdom. Each class consists of between 40 and 258 images, follow this link [http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html]() to the dataset repository to get more detail.

Pre trained CNN Models used:
AlexNet : https://arxiv.org/abs/1404.5997
VGG19 : https://arxiv.org/abs/1409.1556
ResNet : https://arxiv.org/abs/1608.06993

predict.py : program to predict the flower image
cat_to_name.json : json file to map the classes with the flower names
train.py : program to train, validate and test the classifier

Install
Clone the repository to the local machine

$ git clone https://github.com/mohab58977/AI-flower-image-classifier.git

to get help execute:

$ python train.py -h

$ python predict.py -h


Python version:
This app uses Python 3.8.1

Libraries used:
time
numpy
json
PIL
os
torch
torch
torchvision