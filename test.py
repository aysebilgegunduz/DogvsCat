

from __future__ import division, print_function, absolute_import

import numpy as np
import csv
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
from skimage import io
from scipy.misc import imresize

from sklearn.model_selection import train_test_split
import os
from glob import glob


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

# test-train split
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

#################################
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

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 64, 3, activation='relu', name='conv_1')
conv_1 = conv_2d(conv_1, 32, 3, activation='relu', name='conv_11')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')
conv_2 = conv_2d(conv_2, 40, 3, activation='relu', name='conv_22')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')
conv_3 = conv_2d(conv_3, 32, 3, activation='relu', name='conv_33')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

model = tflearn.DNN(network)
model.load('cat_dog_model_final.tflearn')
result = []
logloss = 0
"""
with open('a.csv', 'rb') as tests:
    testy = tests.readlines()

for i in xrange(0, len(testX)):
        im = [testX[i]]
        a = model.predict(im)
        if a[0][0]<a[0][1]:
            result.append(1)
        else:
            result.append(0)
"""
with open('deneme.csv','wb') as cs:
    writer = csv.writer(cs, delimiter=',',quoting=csv.QUOTE_NONE)
    for i in xrange(0, len(testX)):
        im = [testX[i]]
        a = model.predict(im)
        if a[0][0]<a[0][1]:
            writer.writerow([str(i+1), 1])
        else:
            writer.writerow([str(i + 1), 0])

