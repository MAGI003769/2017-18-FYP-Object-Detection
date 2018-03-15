import cv2
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model

from keras.preprocessing import image


# Load image
imgs = []
input_shape = (100, 100, 3)
img_path = './img/shu.jpg'
img = image.load_img(img_path, target_size=(100, 100))
img = image.img_to_array(img)
imgs.append(img)
imgs = np.asarray(imgs)
print(imgs.shape)


# Construct the simple model
'''
All the layers called return a Tensor type that cannot be directly 
visualized by matplotlib
'''
net = {}
input_tensor = Input(shape=input_shape)  
net['input'] = input_tensor
'''
A conv layer expect an input 
with dimension of [batch_size, width, height, channel]
'''
net['conv_1'] = Conv2D(9, (10,10),
	                   #activation='relu',
	                   padding='valid', 
	                   name='conv_1')(net['input'])

net['BN_1'] = BatchNormalization(axis=-1)(net['conv_1'])

# Different output
model_conv1 = Model(net['input'], net['conv_1'])
model_BN = Model(net['input'], net['BN_1'])

# Feed model
conv_out = model_conv1.predict(imgs)
BN_out = model_BN.predict(imgs)

for i in range(conv_out.shape[3]):
	plt.subplot(3,3,i+1)
	plt.axis('off')
	plt.imshow(conv_out[0, :, :, i] / 255. , cmap='Greys')
plt.savefig('results\\only_conv.png')
plt.show()
print(net['conv_1'][0, :, :, :].shape)

for i in range(BN_out.shape[3]):
	plt.subplot(3,3,i+1)
	plt.axis('off')
	plt.imshow(BN_out[0, :, :, i] / 255. , cmap='Greys')
plt.savefig('results\\conv+BN.png')
plt.show()
print(net['BN_1'][0, :, :, :].shape)