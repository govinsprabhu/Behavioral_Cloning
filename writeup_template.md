# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/placeholder.png "Model Visualization"
[image2]: ./images/placeholder.png "Grayscaling"
[image3]: ./images/placeholder_small.png "Recovery Image"
[image4]: ./images/placeholder_small.png "Recovery Image"
[image5]: ./images/placeholder_small.png "Recovery Image"
[image6]: ./images/placeholder_small.png "Normal Image"
[image7]: ./images/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I have used the model that has described in NVIDIA autonomous [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). There is a pre-procesing layer, which normalizes the image, then there are 6 convolution layer and 4 fully connected dense layers are present in the model.  Model can be found in model.py file from line numbers 69 to 119 inside get_number method. 
 I have used RELU for activation, did not use maxpooling. Data is normalized by keras lambda layer.

#### 2. Attempts to reduce overfitting in the model
I have tried to use dropout layer, but I did not found it making much difference. Rather than dropout, I found, the more data we are feeding to the network, the better it peform. So I took data from simulater additional to the one which is given by Udacity.
 #### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Below are the full set of data set I have used for training

 * Two full lap in the network
 * One full reverse lap 
 * Recovery lap for getting car recorverd incase if it is going to side ways

For details about how I created the training data, see the next section. 
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train the car to run through the road without being deviating to the side ways.

My first step was to use a convolution neural network model similar to that of  [NVIDIA paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because they have used this model to train a real car to run through the road autonomously.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Initially the difference between training and validation error was more, so decided to add dropout. But dropout was not helping me much. So I have decided to collect more data.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially in the curves. To improve that, I have taken more data. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

 The final model can be found in model.py file from line numbers 69 to 119 inside get_number method. It consiste of a convolution neural network with the following layers and layer sizes 
 

Here is a visualization of the architecture.  

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I have used data provided by the course to train the model initially, later found out the data was not suffiecient, since the model is suffering from overfitting. So I have added more data. Following additional data I have captured

 * Two full lap in the network 
 * One full reverse lap 
 * Recovery lap for getting car recorverd incase if it is going to side ways
 
 Also, I have loaded left, right data for each of the central images. For each of it, I took the vertical flip of the image for further augmentation. Below are the images. 


![alt text][image6]
![alt text][image7]


After the collection process, and taking its left and right plus their flip images,  I had 96432  number of data points. I then preprocessed this data by using keras lambda layer, normalized it by dividing 255, then subtracting 0.5 from it. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by training and validation error was not reducing further. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I did not trained every data at one go. Step by step, I have trained my model. Initially started with Udacity data, then as I found out the data was insuffiecient, I have added more number of laps manually. After each time, I have saved the model and trained it again for the new set of data. All the model h5 I have cheked. Final model is [model.h5](https://github.com/govinsprabhu/Behavioral_Cloning/blob/master/model_final.h5)
