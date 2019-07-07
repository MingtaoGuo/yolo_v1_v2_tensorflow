# yolo1_tensorflow
Simple implementation of yolo v1 and yolo v2 by TensorFlow

# Introduction
Paper yolo v1: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640)

Paper yolo v2: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)

The code of yolo v2, we use 9 anchors which is calculated by k-means on COCO dataset.

|data augmentation|pretrained vgg16|pretrained darknet|
|-|-|-|
|:x:|:heavy_check_mark:|:x:|
## What is normalized offset?
![](https://github.com/MingtaoGuo/yolo1_tensorflow/blob/master/IMGS/norm.jpg)
# Requirements

==============
1. python3.5
2. tensorflow1.4.0
3. pillow
4. numpy
5. scipy

Pretrained VGG16: Google Drive: [https://drive.google.com/open?id=1LTptCY96ABAUlJHUJq6MhqNrDQN7JfQP](https://drive.google.com/open?id=1LTptCY96ABAUlJHUJq6MhqNrDQN7JfQP)

Dataset: Pascal voc 2007: [https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar](https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar)

==============

# Results

|![](https://github.com/MingtaoGuo/yolo1_tensorflow/blob/master/IMGS/loss.jpg)|![](https://github.com/MingtaoGuo/yolo1_tensorflow/blob/master/IMGS/bbox.jpg)|
|-|-|
|![](https://github.com/MingtaoGuo/yolo1_tensorflow/blob/master/IMGS/ironman.jpg)|![](https://github.com/MingtaoGuo/yolo1_tensorflow/blob/master/IMGS/avg.jpg)|
|![](https://github.com/MingtaoGuo/yolo1_tensorflow/blob/master/IMGS/1.jpg)|![](https://github.com/MingtaoGuo/yolo1_tensorflow/blob/master/IMGS/2.jpg)|
|![](https://github.com/MingtaoGuo/yolo1_tensorflow/blob/master/IMGS/3.jpg)|![](https://github.com/MingtaoGuo/yolo1_tensorflow/blob/master/IMGS/4.jpg)|

# Reference
[1]. Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[2]. Redmon J, Farhadi A. YOLO9000: better, faster, stronger[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7263-7271.
