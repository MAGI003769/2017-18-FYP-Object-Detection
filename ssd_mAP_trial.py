import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import pickle
import os
from os import listdir
from os.path import isfile, join

from ssd import SSD300
from ssd_utils import BBoxUtility
from ssd_metrics import iou_cal, AP

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1


# Initialize pre-trained model and set size of input
input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)
#model.load_weights('./checkpoints/earliest/weights.13-7.92.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

# Path of necessary files
test_img_prefix = 'E:/PASCAL_VOC_Dataset/VOC2012test/VOCdevkit/VOC2012/JPEGImages'
test_data_path = 'E:/PASCAL_VOC_Dataset/VOC2012_test.pkl'

# Get ground truth data and keys: dict{'key': (n, 24), ...}
all_ground_truth = pickle.load(open(test_data_path, 'rb'))
all_keys = sorted(all_ground_truth.keys())
num_test = 100
select_keys = all_keys[:num_test]
print(select_keys)

# Get images for selected keys
inputs = []
images = []
images_path = [join(test_img_prefix, key) for key in select_keys if isfile(join(test_img_prefix, key))]
for i in range(len(images_path)):
	img_path = images_path[i]
	img = image.load_img(img_path, target_size=(300, 300))
	img = image.img_to_array(img)
	images.append(imread(img_path))
	inputs.append(img.copy())

# Employ pretrained model to predict
inputs = preprocess_input(np.array(inputs))
preds = model.predict(inputs, batch_size=1, verbose=1) # (num_test, 7308, 33)
results = bbox_util.detection_out(preds)               # num_test arrays (num_det, 6)

average_precision = []
for i in range(num_test):
	print('\n======== Test sample {} =========\n'.format(i))
	print('\nresult: {}\n'.format(results[i]))
	print('ground_truth: {}\n'.format(all_ground_truth[select_keys[i]]))
	temp_AP = AP(results[i], all_ground_truth[select_keys[i]])
	average_precision.append(temp_AP)

average_precision = np.asarray(average_precision)
average_precision = np.sum(average_precision, axis=0)
print(average_precision)
