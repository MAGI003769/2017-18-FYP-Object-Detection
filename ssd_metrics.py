import tensorflow as tf
import numpy as np


def iou_cal(y_pred, y_true):
	''' Function to calculate IOU of prediction and ground truth
	Arguments
	---------
	y_pred: pridiction results (filted by conf) with size (num_box, 6)
	y_true: ONE ground truth (24, ), [:4] are coodinates while [4:] are one-hot labels

	Return
	------
	iou of predictions and ground truth (num_box, )

	'''

	# Compute intersection
	inter_upleft = np.maximum(y_true[:, :2], y_pred[2:4])
	inter_botright = np.minimum(y_true[:, 2:4], y_pred[4:6])
	inter_wh = inter_botright - inter_upleft
	inter_wh = np.maximum(inter_wh, 0)
	inter = inter_wh[:, 0] * inter_wh[:, 1]
	# Computer union
	area_pred = (y_pred[4] - y_pred[2])*(y_pred[5] - y_pred[3]) 
	area_gt = (y_true[:, 2] - y_true[:, 0])*(y_true[:, 3] - y_true[:, 1])
	union = area_gt + area_pred - inter
	# iou
	iou = inter / union
	return iou

def AP(y_pred, y_true, iou_thresh=0.5):
	average_precisions = []

	num_class = y_true.shape[1]-4
	for index in range(num_class):
		print('\n class: {} \n'.format(index))
		TP, FP = 0, 0
		class_index_truth = np.where(y_true[:,index+4]==1)
		class_index_pred = np.where(y_pred[:, 0]==index+1)
		if class_index_truth[0].shape[0] == 0:
			average_precisions.append(0)
			continue
		if class_index_pred[0].shape[0] == 0:
			average_precisions.append(0)
			continue
		#print('!!!')
		# Compare each prediction with ground truth boxes
		for i, detect in enumerate(class_index_pred[0]):
			#print('class_index_pred: ', class_index_pred)
			iou = iou_cal(y_pred[detect], y_true)
			print('\niou: ', iou)
			if iou.max() > iou_thresh:
				TP += 1
			else:
				FP += 1
		average_precisions.append(TP / (TP+FP))

	#mAP = sum(average_precisions) / len(average_precisions)
	return average_precisions#, mAP

def recall(y_pred, y_true):
	recalls = []

	num_class = y_true.shape[1] - 4
	for index in range(num_class):
		TP, FN = 0, 0
		class_index_truth = np.where(y_true[:,index+4]==1)
		class_index_pred = np.where(y_pred[:, 0]==index)
		if len(class_index_pred)==0:
			FN += len(class_index_truth)
		else:
			for i, detect in enumerate(class_index_pred):
				iou = iou_cal(y_pred[detect], y_true)
				if iou.max() > iou_thresh:
					TP += 1
		recalls.append(TP/(TP+FN))

