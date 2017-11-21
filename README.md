# 2017-18-FYP-Object-Detection

My undergraduate final year project, supervised by [Dr. Qiufeng Wang](http://www.xjtlu.edu.cn/zh/departments/academic-departments/electrical-and-electronic-engineering/staff/qiufeng-wang), which intends to implement object detection in images using SSD method. This repository, whose `README.md` is a record of learning outcome and experiment observation, contains the paper, implementation code and relevant data. 

The detail theory refers to the [arXiv paper](https://arxiv.org/abs/1512.02325). Initially, I read and understand the methodology in paper with a [keras version](https://github.com/rykov8/ssd_keras) implementation. Considering about current stage that I still a tyro in this field, it is most likely to make modifications on such a version based on user-friendly interface. For better performance, it may be better to refer a [tensorflow version](https://github.com/balancap/SSD-Tensorflow) with more advanced features. With the study goes further, more methodologies and techniques come into view. 

# Comprehension of Methodology

This part essentially record the cues of my study on this project. There will be some links to access papers, technique blogs referred through out my study. As well, some notes written to record my understanding and learning outcomes on specific topics  that relate to this topic will be provided.

## Recognition Network (Base Network)

Most methodologies used for object detection has a network to **_extract features_** of different level and then recognition. Such a recognition can be regarded as a sub-task for object detection since we need to know the existence of objects before we locate them (or in other word label them with bounding boxes). Convolutional Neural Networks (CNN) has proved their conspicuous performance in recognition tasks on images. In current stage of my study, I make some notes on following network architectures:

- VGG
- [ResNet](Papers and Notes/Residual Learning.md)

## Region Related

This project employs **_Bounding Boxes_** to show the results of object detection rather than **_segmentation_** which is a much more advanced task. 

- Region of Interest (RoI)


- Region Proposal Network (in Faster R-CNN)
- Anchor

## Integrated Detection Methodologies

Even though, as previously mentioned, this project mainly study and implement SSD for object detection. It is improper to ignore the methodologies which perform well in this task. Following list collects some remarkable methodologies.

- SSD
- Faster R-CNN
- YOLO (YOLO9000)
- R-FCN (Fully-Convolutional Network)
- SSPNet (Spatial pyramid pooling)
- Mask R-CNN (This is optional one as it is for advanced segmentation)

# Record of Study, Experiment and Observation

## 2017/9/27

- Add more testing images. The accuracy is, although not bad, lower than expected. Particularly, some objects are ignored as their low confidence. There is, as well, some wrong classification (eg. treat horse as cow).
- ~~There was no dropout for current trained network. Thus, it is conjectured as the reason for not sufficiently good result.~~  After further study, I think the dropout regularization exists when training the network.

## 2017/10/7

- Study of SSD architecture: Based Network
- Learning TensorFlow frame work

## 2017/10/17

- Review of CNN to help understanding
- Review some basic of Deep Learning

##2017/10/31

- Learning of Keras interface
- Code reading and understanding about paper

## 2017/11/8

- Study of training methods and loss function of SSD
- _**Question**: Confusions on number of bbox selection_

## 2017/11/15

- Study of **_Deep Residual Learning_** which is a very deep architecture but easy to train. 
- This is study to explore the possibility of change the recognition network (or base network) for a higher confidence score or better performance on small objects.

## 2017/11/21

- More details on training methodology. 
- _**Question**: The classifier for each feature map is just a convolutional layer and then flattened layer without fully-connection. So, is it not necessary for a classifier to have FC?_ [The function of FC](https://stats.stackexchange.com/questions/182102/what-do-the-fully-connected-layers-do-in-cnns)
- About different scale feature map: It is initially confusing to make the b-boxes of different feature layers  correctly plotted on the images. But, through the reading of code, I observed that the shape of those layers are possibly the same. Pooling layers???

# Web Source for Reference

- [Zhihu - Why is SSD weak in small object detection?](https://www.zhihu.com/question/49455386)
- [Zhihu - What improvements does SSD method have?](https://www.zhihu.com/question/50910763)