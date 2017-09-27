# 2017-18-FYP-Object-Detection

My undergraduate final year project, supervised by [Dr. Qiufeng Wang](http://www.xjtlu.edu.cn/zh/departments/academic-departments/electrical-and-electronic-engineering/staff/qiufeng-wang), which intends to implement object detection in images using SSD method. This repository, whose `README.md` is a record of learning outcome and experiment observation, contains the paper, implementation code and relevant data. 

The detail theory refers to the [arXiv paper](https://arxiv.org/abs/1512.02325). Initially, I read and understand the methodology in paper with a [keras version](https://github.com/rykov8/ssd_keras) implementation. Considering about current stage that I still a tyro in this field, it is most likely to make modifications on such a version based on user-friendly interface. For better performance, it may be better to refer a [tensorflow version](https://github.com/balancap/SSD-Tensorflow) with more advanced features. 

## Comprehension of Methodology

## Experiment Record and Observation

### 2017/9/27

- Add more testing images. The accuracy is, although not bad, lower than expected. Particularly, some objects are ignored as their low confidence. There is, as well, some wrong classification (eg. treat horse as cow).

- There was no dropout for current trained network. Thus, it is conjectured as the reason for not sufficiently good result.
