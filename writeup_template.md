# **Behavioral Cloning** 

## Project Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/cropped_image.png "Image preprocessing"
[image3]: ./examples/camera_angles.png "All used cameras"
[image4]: ./examples/recovery_maneuver.png "Recovery Image"
[image5]: ./examples/flipped_image.png "Image flip"
[image6]: ./examples/angle_correction.png "Angle correction"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network, the LeNet-5
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
using the provided trained LeNet-5 CNN that is included in this submission.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with something rather simple,
test it on an initial dataset and gradually introduce complexity that improves the result on this
small dataset until the car is able to successfully drive some distance. 

As a first step a very simple architecture of a flattening and two fully connected layers of sizes
300 and 100, that end in a single output node, was used. 

In order to test the model, a first round of centered driving was recorded and I subsequently the
camera images were the input for training, while the angle data was the output for backpropagation.
It was found that the first model had a low mean squared error on the training set but a high mean squared
error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, a dropout layer was introduced and the training epochs were adjusted accordingly.

After the training the model was used to predict angles during PID controlled speed on the first
track, i.e. a fully autonomous test drive. Here it was observed that the model performed very poorly
and almost immediately drove off the track.

The next step to improve the performance was to collect more data. After another training session, 
the model clearly performed better, but was not able to drive very far before driving off the track.

For this reason, a much more suitable architecture, the LeNet-5 was chosen, as it was shown to work,
very well with pattern recognition, which is the gist of what we want to achieve here.

After some architecture adjustments, all described below in detail and more data collection,
the vehicle is able to drive the track autonomously without ever leaving the track and is able to
stay centered in the lane most of the times. There are a few spots where the vehicle comes rather
close to the road edge, and there are oscillations in the steering, which is behaviour that we 
ideally would not want in our model, yet much more debugging and training would have to be done
to completely get rid of these features.

#### 2. Final Model Architecture

The neural network architecture pipeline is built in the function
```
create_model_architecture(input_shape=(160, 320, 3))
```
in  model.py (lines 20 - 44), and expects arrays of dimensions (160, 320 , 3) as an input (images
with three color channels).

Before the input into the neurons, there is a data processing block using keras lambda layers.
It normalizes the intensity
of each pixel of range (0, 255) and subsequently centers around zero, such that each color channel
of each pixel now has a range of (-0.5, 0.5). Furthermore the upper 50 pixel rows and the lower 20
pixel rows are cutoff to get rid of the unnecessary sky and car hood (which implies that we assume
to only be driving in a flat world). This is shown in the image below:

![alt text][image2]

on the left and exemplary input camera image is shown, while on the left the input to the first
convolutional layer is depicted.

Subsequently, the LeNet-5 architecture follows, as first described by Yann LeCun et al.
[LeCun, Y.;
Bottou, L.; Bengio, Y. & Haffner, P. (1998).
Gradient-based learning applied to document recognition.
Proceedings of the IEEE. 86(11): 2278 - 2324]
Which consists of:
* A convolution layer using 6 kernels with a size of 3x3 using the ReLU activation function
* An average pooling layer with a pooling size of 2x2
* A convolution layer with 16 kernels and with a size of 3x3 using the ReLU activation function
* An average pooling layer with a pooling size of 2x2
* A flattening layer and a subsequent dropout layer that randomly drops 20% of the data to prevent
overfitting
* A fully connected layer with 120 neurons and ReLU activation
* A fully connected layer with 60 neurons and ReLU activation
* A fully connected layer with 1 neurons and no activation function as it's output is the steering
angle

Certainly there can be more improvements wihtin this model, that have not been explored yet, as for
example an adjustment of convolutional kernel sizes, to capture large patterns better, or additional
convolutional layer within the network. Moreover another architecture could have been chosen, while
the convolutional approach seems to be the most appropriate one overall.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one counter-clockwise lap on track one using
center lane driving.
An example image of all three cameras during such a lap is shown in the image below:

![alt text][image3]

As it was quite difficult to achieve the desired center lane driving in the simulator during all 
time and to have a more consistent dataset, another three laps of counter-clockwise driving and
subsequently another three laps of clockwise center lane driving were recorded.

It was then found, that the vehicle still had a tendency to maneuver off track when it got too close
to either side of the track. To tackle this problem, three rounds of recovery maneuvers from the
edge of the road back to the center were recorded, equally distributed on both sided of the road.
An example of such a recovery maneuver is shown below in four equally spaced timesteps.

![alt text][image4]

Importantly, the recording was turned of during the approach of the road edge, to not train the
model to drive towards the road edge.

To enhance the data set, different things were tested, for example inverting the images on the
y-axis and negating the corresponding steering angle, as shown on the image below:

![alt text][image5]

which doubles the size of the dataset.

Furthermore, each recording actually consists of two additional cameras to the left and the right
of the vehicle, which can also be utilized for training. However, the steering angles for these
cameras have to be adjusted, as they are mounted in different positions. Through trial and error 
a correction factor of 0.3° was found to be suitable, which was then used for training. The three
camera images with their corrected angles at one timestamp are shown in the image below.

![alt text][image6]

Yet, it was found that using both data enhancement methods, flipping and all cameras, actually
achieves a bit worse performance than using just one of them and in the final model, no flipping
of camera images has been used (see the useage of images and angles instead of augmented_images and
augmented_angles in line 80 and 81 of model.py).
This could for example stem from a bad correction angle.

After the collection process, the final dataset consists of 51204 images and their corresponding
steering angles. The sheer size of this data drives the RAM capacity allocated to many python
environments to or beyond their limits, and so the data collection and readout was implemented using
a python generator, that reads out and stores only 32 images at a time in the RAM.
The implementation is done in the function 
```
generator(samples, batch_size=32)
```
in lines 47-82 in model.py, where also the angle correction and (not used) image inversion can be
found.

Finally the data was randomly shuffled 20% of the data were put into the validation set. 

The loss function used for the training to optimize on was the mean squared error and as an adam
optimizer was used, it was not necessary to tune the learning rate of the training.
The loss on the validation set was utilized to determine if the model was over or under fitting and 
accordingly the ideal number of epochs was determined to be 7 as evidenced by no more change in loss
on training or validation set (lines 111 -116 in model.py).

