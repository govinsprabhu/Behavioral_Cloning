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

[image2]: ./images/cnn-architecture-624x890.png "Nvidia model"
[image3]: ./images/sample_dataset.png "Center, left, right images with their flips"
[image4]: ./images/architecture.png "My final Architecture"
[image5]: ./images/SampleImge.png "Sample Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. The submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_final.h5 containing a trained convolution neural network (changed the name for clarity)
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
I have used the model that has described in NVIDIA autonomous [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). There is a pre-processing layer, which normalizes the image, then there are 6 convolution layer and 4 fully connected dense layers are present in the model.  The model can be found in model.py file from line numbers 69 to 119 inside get_number method. 
 I have used RELU for activation, did not use maxpooling. Data is normalized by keras lambda layer.

#### 2. Attempts to reduce overfitting in the model
I have tried to use dropout layer, but I did not find it makes much difference. Rather than dropout, I found, the more data we are feeding to the network, the better it performs. So I took data from simulator additional to the one which is given by Udacity.
 #### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Below are the full set of data set I have used for training

 * Two full lap in the network
 * One full reverse lap 
 * Recovery lap for getting car recovered in case if it is going to sideways

For details about how I created the training data, see the next section. 
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train the car to run through the road without being deviating to the sideways.

My first step was to use a convolution neural network model similar to that of  [NVIDIA paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because they have used this model to train a real car to run through the road autonomously. Below is the visualization of the architecture. I have used exactly the same architecture, with the same filter, but different size of the input image

![image2]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Initially, the difference between training and validation error was more, so decided to add dropout. But dropout was not helping me much. So I have decided to collect more data.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially in the curves. To improve that, I have taken more data. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

 The final model can be found in model.py file from line numbers 69 to 119 inside get_number method. It consists of Normalization layer, followed by 6 convolutions neural network with the following 4 fully connected layers.
 
Here is the summary of the architecture (Not visualization)

![image4]

#### 3. Creation of the Training Set & Training Process

I have used data provided by the course to train the model initially, later found out the data was not sufficient since the model is suffering from overfitting. So I have added more data. Following additional data, I have captured

here is a sample image of the center view
![image5]

 * Two full laps in the network 
 * One full reverse lap 
 * Recovery lap for getting car recovered in case if it is going to sideways
 
 Also, I have loaded left, right data for each of the central images. For each of it, I took the vertical flip of the image for further augmentation. Below are the images. 


![image3]

I have not mixed all of the data together, rather, I have first trained the network with data provided by Udacity. After adding left, right and flip for all three, there were 48216 data points were there.

I split the dataset of training purpose into training and validation, put 20% of the data into a validation set. 

I used this 80% data for training the model. The validation set helped determine whether the model overfitting or underfitting. The ideal number of epochs was 2 as evidenced by training and validation error was not reducing further. 
But the model was not able to drive Udacity data along, so I have added two full laps of data from the track. I trained the network for 2 epochs with it.
When finding out this data was too not sufficient, trained with the reverse track and recovery track. After that, model able to predict the steering angle well

I used an Adam optimizer so that manually training the learning rate wasn't necessary.

After each timeI trained, I have saved the model and trained it again for the new set of data. All the model files h5 I have checked in. Final model is [model_final.h5](https://github.com/govinsprabhu/Behavioral_Cloning/blob/master/model_final.h5)
