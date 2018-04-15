import tensorflow as tf
import numpy as np

def iou_cal(y_pred, y_true):
	''' Function to calculate IOU of prediction and ground truth

	# Arguments

	    y_pred: ONE Prediction result (filted by conf), shape: (6, )
	    y_true: Ground truth results, shape: (num_obj, 5)

	# Return

	iou of predictions and ground truth
	'''
	# Computer intersection
	if y_true.shape[0]==0:
		iou = np.array(0)
		return iou
	y_pred[2:4] = np.maximum(y_pred[2:4], np.array([0,0]))
	#y_pred[4:6] = np.minimum(y_pred[4:6], img_shape)
	inter_upleft = np.maximum(y_pred[2:4], y_true[:, 1:3])
	inter_botright = np.minimum(y_pred[4:6], y_true[:, 3:5])
	inter_wh = inter_botright - inter_upleft
	inter_wh = np.maximum(inter_wh, 0)
	inter = inter_wh[:, 0] * inter_wh[:, 1]

	# Computer union
	area_pred = (y_pred[4] - y_pred[2]) * (y_pred[5] - y_pred[3])
	area_true = (y_true[:, 3] - y_true[:, 1]) * (y_true[:, 4] - y_true[:, 2])
	union = area_pred + area_pred - 1.2*inter
	# IOU
	iou = inter / union
	return iou


def iou(y_pred, y_true):
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


