import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from data_generator.object_detection_2d_data_generator import DataGenerator
from ssd_metrics import iou_cal

# .txt files prefix and class names
txt_prefix = './txt_results/ssd300_07+12_2007_test_eval/comp3_det_test_'
img_prefix = 'E:/PASCAL_VOC_Dataset/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# Parameter setting
confidence_thresh = 0.4
iou_thresh = 0.45

# Get ground truth data from pickle files. 
labels = pickle.load(open('VOC2007_test_labels.pkl', 'rb')) # 3D list, int type: (4952, num_boxes, 5)
labels = [np.asarray(label) for label in labels]
ids = pickle.load(open('VOC2007_test_ids.pkl','rb'))                     # 1D list, str type: (4952, )
ids = np.asarray(ids, dtype=np.float)
keys, values = ids, labels
ground_truth = dict(zip(keys, values))

average_precision = []
average_falsePos = []

for i in range(1,21):
	### Reset TP and FP for each class
	TP, FP= 0, 0

	### Read result from text file of a class
	file_path = txt_prefix + classes[i] + '.txt'
	detections = pd.read_csv(file_path, sep=' ', header=None)
	#print(detections)
	detections = detections.as_matrix()                           # predictions (num_detec, 6): [img_ID conf xmin ymin xmax ymax]
	filted_index = np.where(detections[:, 1] > confidence_thresh) # filtered detections
	filted_detec = detections[filted_index]
	image_ids = np.unique(filted_detec[:, 0]).tolist()
	
	### For each detection, find the corresponding image and ground truth of same class
	for j, detec in enumerate(filted_detec):
		key_temp = detec[0]
		gt_temp = np.asarray([ gt for gt in ground_truth[key_temp] if gt[0]==i ])
		#img_shape_temp = cv2.imread(img_prefix+'{:06d}.jpg'.format(int(key_temp))).shape[:2]
		'''
		print("\n================\n")
		print('gt_temp: ', gt_temp)
		print('one_pred: ', detec)
		'''
		ious = iou_cal(detec, gt_temp)
		if ious.max() > iou_thresh:
			TP += 1
		else:
			FP += 1
	
	### Calculate AP
	ap_temp = TP / (TP+FP)
	print('\nAP of class '+classes[i]+' is: ', ap_temp)
	average_precision.append(ap_temp)

mAP = sum(average_precision) / len(average_precision)
print("\nThe mAP is: ", mAP)





