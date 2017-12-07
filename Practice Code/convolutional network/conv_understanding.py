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
	                   activation='relu',
	                   padding='valid', 
	                   name='conv_1')(net['input'])

net['BN_1'] = BatchNormalization(axis=-1)(net['conv_1'])

net['pool_1'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
	                         name='pool_1')(net['conv_1'])

#net['pool_1'] = net['conv_1']

net['conv_2'] = Conv2D(9, (10,10),
	                   padding='valid',
	                   name='conv_2')(net['pool_1'])

model_inter = Model(net['input'], net['conv_1'])
model_pool_1 = Model(net['input'], net['pool_1'])
model_general = Model(net['input'], net['conv_2'])

# Feed input tensor to model
output_inter = model_inter.predict(imgs)
output_pool_1 = model_pool_1.predict(imgs)
output_general = model_general.predict(imgs)

print(net['conv_1'].shape)
print(type(net['pool_1'][0, :, :, :]))
print(type(output_general[0, :, :, :]))

for i in range(output_inter.shape[3]):
	plt.subplot(3,3,i+1)
	plt.axis('off')
	plt.imshow(output_inter[0, :, :, i] / 255., cmap='Greys')
plt.savefig('results\\conv_1_16_(5,5).png')
plt.show()

print(net['conv_1'][0, :, :, :].shape)
#pool_1_ = net['pool_1']
#print(pool_1_.shape)

for i in range(output_pool_1.shape[3]):
	plt.subplot(3,3,i+1)
	plt.axis('off')
	plt.imshow(output_pool_1[0, :, :, i] / 255. , cmap='Greys')
plt.savefig('results\\pool_1.png')
plt.show()
print(net['pool_1'][0, :, :, :].shape)


for i in range(output_general.shape[3]):
	plt.subplot(3,3,i+1)
	plt.axis('off')
	plt.imshow(output_general[0, :, :, i] / 255. , cmap='Greys')
plt.savefig('results\\conv_2_16_(5,5).png')
plt.show()
print(net['conv_2'][0, :, :, :].shape)