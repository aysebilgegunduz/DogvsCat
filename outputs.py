

from __future__ import division, print_function, absolute_import
import matplotlib as plt
import numpy as np

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
from skimage import color, io
from scipy.misc import imresize

from sklearn.model_selection import train_test_split
import os
from glob import glob
files_path = 'dataset/test'
cd_files_path = os.path.join(files_path, '*.jpg')

cat_files = sorted(glob(cd_files_path))

n_files = len(cat_files)
print(n_files)
size_image = 64
allX = np.zeros((n_files, size_image, size_image,3), dtype='float64')
ally = np.zeros(n_files)

im = allX[102:103]
plt.axis('off')
plt.imshow(im[0].astype('uint8'))
plt.gcf().set_size_inches(2, 2)

# run images through 1st conv layer
m2 = tflearn.DNN(conv_1, session=model.session)
yhat = m2.predict(im)

# slice off outputs for first image and plot
yhat_1 = np.array(yhat[0])

def vis_conv(v,ix,iy,ch,cy,cx, p = 0) :
    v = np.reshape(v,(iy,ix,ch))
    ix += 2
    iy += 2
    npad = ((1,1), (1,1), (0,0))
    v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
    v = np.reshape(v,(iy,ix,cy,cx))
    v = np.transpose(v,(2,0,3,1)) #cy,iy,cx,ix
    v = np.reshape(v,(cy*iy,cx*ix))
    return v

#  h_conv1 - processed image
ix = 64  # img size
iy = 64
ch = 32
cy = 4   # grid from channels:  32 = 4x8
cx = 8
v  = vis_conv(yhat_1,ix,iy,ch,cy,cx)
plt.figure(figsize = (12,12))
plt.imshow(v,cmap="Greys_r",interpolation='nearest')
plt.axis('off');