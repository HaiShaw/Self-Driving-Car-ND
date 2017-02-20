# Self-Driving-Car-ND

## Repo for Udacity Self-Driving Car Nanodegree Projects


### Project 2: Use Deep Neural Networks and TensorFlow to Build Traffic Signs Classifier
#### Description:
In the autonomous car industry, computer vision and deep learning have many important use cases to both provide accurate scene or context understanding to driving safety and intelligent augmentation or automation to the vehicle control, examples range from lane lines recognition & tracking, pedestrian classification, vehicle detection and tracking or traffic sign classification, etc.
With applications in demand and rapid evolving of GPU, image processing method, architecture, and algorithm deep learning infrastructure software have made great advancement in the recent years, popular ones including Caffe, Torch, Keras, Theano and TensorFlow.
In this project, I use TensorFlow to develop traffic signs image classifiers in addition to traditional machine learning libraries such as sklearn, using Python 3


### Project 3: Use Deep Learning (CNN) and Keras (TensorFlow backend) to Predict Steering Angles and Throttle (optional)
#### Description:
In this project, I use deep neural networks and convolutional neural networks to clone driving behavior. I train, validate and test a model using Keras with TensorFlow backend. The model outputs a steering angle to an autonomous vehicle.
I use Unity simulator from Udacity to drive car around a track for data collection. After that I use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.
Immediately approachable solution: Learn extra driving control signals such as 'throttle' using additional regressor outputs.
##### Project Video: https://youtu.be/VhmtxuBt5GM


### Project 4: Computer Vision Advanced Lane Lines Tracking
#### Description:
In this project, the goal is to write a software pipeline to identify the lane boundaries in a video. Extensive computer vision processings are applied to this video pipeline, ranging from distortion correction, color transform and gradients threshold to 'birds-eye view' transform, lane line pixels histogram exploration and intelligent search, and curvature fit .... Prior to video pipeline, camera is calibrated with calibration images. At video output, visual display of the lane boundaries and numerical estimate of lane curvature and vehicle relative position to the lane center are provided.
##### Project Video: https://youtu.be/tcUFizM5eNU


### Project 5: Computer Vision and Supervised Learning Vehicle Detection and Tracking
#### Description:
In this project, the goal is to write a software pipeline to detect and tracking on road vehicles. First investigate the given vehicle and non-vehicle data set and use computer vision techniques to generate solid feature maps in producing a robust classifier model from supervised learner (LinearSVC considering the speed), then in the video pipeline a same feature extraction and normalization are applied with smart sliding window search deployed to scale. At the end, heatmap average and threshing are applied to further lower the false positives.
Immediate reflection: try different approaches, e.g. haar cascades, deep learning for object detection; or harness multiple camera views and perspective transform to extract 3D euclidean vector (distance) to tracking objects for better real value.
Immediately approachable solution: Add Vehicle Class for active tracking and noise rejection.
##### Project Video: https://youtu.be/fkXnezlYH_I



