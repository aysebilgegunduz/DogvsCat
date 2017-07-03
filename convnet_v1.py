from __future__ import division, print_function, absolute_import

import csv

from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.model_selection import train_test_split
import os
from glob import glob
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
###################################
### Import picture files
###################################

files_path = 'dataset/train'
cat_files_path = os.path.join(files_path, 'cat*.jpg')
dog_files_path = os.path.join(files_path, 'dog*.jpg')
test_files_path = os.path.join('dataset/test1', '*.jpg')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))
test_files = sorted(glob(test_files_path))

n_files = len(cat_files) + len(dog_files)
test_n = len(test_files)
print(test_n)
size_image = 64
allX = np.zeros((n_files, size_image, size_image,3), dtype='float64')
ally = np.zeros(n_files)
testX = np.zeros((test_n, size_image, size_image, 3), dtype='float64')
count = 0
for f in test_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image,size_image,3))
        testX[count] = np.array(new_img)
        count += 1
    except:
        continue
count = 0
for f in cat_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image,size_image,3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue
for f in dog_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image,size_image,3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue

###################################
# Prepare train & test samples
###################################

# test-train split
X, X_test, Y, Y_test = train_test_split(allX,ally,test_size=0.1, random_state=42)

#encode the Ys
Y = to_categorical(Y,2)
Y_test = to_categorical(Y_test, 2)

###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, [3,3], activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, [3,3])
# 12: Dropout layer to combat overfitting
network = dropout(network, 0.8)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, [3,3], activation='relu', name='conv_2')

# 2: Max pooling layer
network = max_pool_2d(conv_2, [3,3])
# 12: Dropout layer to combat overfitting
network = dropout(network, 0.8)

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(network, 64, [3,3], activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, [3,3])
# 12: Dropout layer to combat overfitting
network = dropout(network, 0.8)

# 5: Convolution layer with 64 filters
conv_4 = conv_2d(network, 128, [3,3], activation='relu', name='conv_4')

# 6: Max pooling layer
network = max_pool_2d(conv_4, [3,3])
# 12: Dropout layer to combat overfitting
network = dropout(network, 0.8)

# 7: Convolution layer with 64 filters
conv_5 = conv_2d(network, 256, [3,3], activation='relu', name='conv_5')

# 8: Max pooling layer
network = max_pool_2d(conv_5, [3,3])
# 12: Dropout layer to combat overfitting
network = dropout(network, 0.8)

# 9: Convolution layer with 64 filters
conv_6 = conv_2d(network, 256, [3,3], activation='relu', name='conv_6')

# 10: Max pooling layer
network = max_pool_2d(conv_6, [3,3])

# 12: Dropout layer to combat overfitting
network = dropout(network, 0.8)

# 11: Fully-connected 512 node layer
network = fully_connected(network, 1024, activation='relu')


# 13: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_cat_dog_6.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')

###################################
# Train model for 100 epochs
###################################

model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=200,
      n_epoch=100, run_id='cat_dog_model', show_metric=True) #100 olmali test icin 1'e cektim

model.save('cat_dog_model_final.tflearn')
# choose images & plot the first one
result = []
logloss = 0

with open('deneme.csv','w') as cs:
    writer = csv.writer(cs, delimiter=',',quoting=csv.QUOTE_NONE)
    for i in range(0, len(testX)):
        im = [testX[i]]
        a = model.predict(im)
        if a[0][0]<a[0][1]:
            writer.writerow([str(i+1), 1])
        else:
            writer.writerow([str(i + 1), 0])

