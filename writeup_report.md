
# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center_lane.jpg "Center Lane Driving"
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/flip.png "Flip Image"
[image7]: ./examples/loss.png "MSE Loss"

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

After trying a couple small custom architectures based off of the original single-layer presented, I tried the NVIDIA Self-driving architecture from [the recommended paper](https://arxiv.org/pdf/1604.07316.pdf). I ended up adding a single dropout layer between the convolutional network and fully-connected layers.

The model consists of a convolution neural network with 5x5 filter sizes for depths 24, 36, 48, followed by 3x3 convolutions for depths 64 and 64 (model.py lines 66-70). All convolutional layers have rectified linear unit activation functions for non-linearity and mitigating the vanishing gradient problem. The 5x5 convolutional layers have an additional subsample routine of size 2x2 (model.py lines 66-68).   

The data is normalized in the model via the max element and zero-centering with 0.5 using a Keras lambda layer (model.py line 64). Afterwards, the input image/tensor is cropped to 70 width and 25 height (model.py line 65).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 71). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 14-33, 79). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 78).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the circuits backwards, and targetted bridge, dirt and turning situations. 

For details about how I created the training data, see the next section. 

#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to resemble human-driving with the simplest architecture possible.

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because it has performed well in on actual streets, though with much more data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that 50% of the information from the convolutional layers is dropped.

Then I feed through FC layers of 1164 (Flatten), 100, 50, and 10 nodes, ending at a single node outputing the inverse turn radius.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as dirt, turns, and elevation. To improve the driving behavior in these cases, I gathered more data, recovering from these situations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 6. Final Model Architecture

Model architecture described in 1.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 7. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded some laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to navigate away from the borders. These images show what a recovery looks like starting from the right side of a left turn:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To gather a different perspective and correct the left bias, I trained on the reverse of the circuit, making sure I began recording when I've turned around. Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles. Trying without, this augmentation seemed to help the directional invariants:

![alt text][image6]


After the collection process, I had 20840 number of data points. I then preprocessed this data by normalizing and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the below loss graph. My circuit2 run was from a model trained over 7 epochs. A graph of 4< epochs showed that my loss generally increased after 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image7]
