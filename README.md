# FaceDetection
A CNN model trained to distinguish five different people

Copyright (C) 2018 Jingke Zhang 
Tsinghua University, China
zjk17@mails.tsinghua.edu.cn

## Setup
Before you start the lab, you should first install:
* Python 3
* TensorFlow
* NumPy
* SciPy
* matplotlib

## Important
Please put /V1.0/CNN.py under the root path of this project.

## Notice
There is a tiny thing you need to do after downloading the code and begin to run it. In order to save the network parameter for reuse, during the training, model parameters will be saved in path: './Model_saved'. tensorboard records: './test' haven't been uploaded to github. So please create these two folder in your root path, and enjoy the code! 

## Version explanation

In V0.0 version, I tried to follow the instruction on the book: "TensorFlow for Machine Intelligence", which suggested me to use TFRecord file to organize my training data and labels. I understand it can ficilitate the training process: no need to match data and labels during the training anymore. Even so, I gave up this way and developed V1.0 which doesn't use TFRecord anymore but only loading a given number of files to make a batch during the training. 

To be honest, this is a pretty simple classification problem: only five person to distinguish. However, the CNN Network in this model still consist of 5 convolutional layers and two fully-connected layers. Actually, it's because when using a 2+2 Network, as the training times increase, the output of the CNN became all zeros, which confused me. The activation function used in the Network is ReLU, it is said the ReLU Network could perform better in a pretty deep Network, so I switched to the configuration I'm using now, and it seems like it can work well.

One more thing: In the function 'image_load_display_preprocessing()': image.open() was called to load in the image data very frequently, about 30 times per training loop. In the begining, I used plt.show() to visualize the training process, which always led to corruption of Pycharm at around 70-80th training loop. I suppose it's because the overmuch images showed in Pycharm cost a huge amount of memory which exceed the memory limit set by system.

## Accuracy
In the begining of the training process, all outputs of the network were 0, and this status continued for about 200 batches(20 images in one batch). After 400 training loops, the cost function began to decrease, and after 1000 loops reach about almost proximate 0. 
For evaluating the accuracy of the classification, two test dataset have been provided. One: 20 images included in training dataset. Two: 20 images not included in training dataset. In the early stage of the training, accuracy of the classification of these two dataset are both rather low: 0.1-0.2. The training has continued for about 2000 round, by the time I stopped it, the accuracy of classificaition of dataset one reach 1.0, with accuracy of two still fluctuate around 0.5. 
In conclusion, the CNN can work properly with unsatisfied accuracy of classification of the new image. But I think it's not owing to the deficiency of the network or the architecture. It's the lack of training data that cause this outcome. And obviouly the network was overfit, maybe we can increase the percentage of drop out layer to overcome it. 
                                                                                                                  8/10/2018
