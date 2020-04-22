# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Data_Distribution.png "Distribution"
[image2]: ./examples/Examples_for_Labels.png "Visualization"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images_from_web/img0.jpg "Traffic Sign 1"
[image5]: ./images_from_web/img1.jpg "Traffic Sign 2"
[image6]: ./images_from_web/img2.jpg "Traffic Sign 3"
[image7]: ./images_from_web/img3.jpg "Traffic Sign 4"
[image8]: ./images_from_web/img4.jpg "Traffic Sign 5"
[image9]: ./examples/HistEqualization.jpg "Histogram equalization step"
[image10]: ./examples/Preprocessed.jpg "Preprocessing step"
[image11]: ./examples/WebImages.png "Top Softmax Probabilities"
[image12]: ./examples/Accuracy_History.png "History Accuracy"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used pythons standard library as well as numpys library to calculate summary statistics of the traffic signs data set:

* The size of training set is `len(X_train)`
* The size of the validation set is `len(X_valid)`
* The size of test set is `len(X_test)`
* The shape of a traffic sign image is `X_train[0].shape`
* The number of unique classes/labels in the data set is `len(np.unique(y_train))`

#### 2. Include an exploratory visualization of the dataset.

I created three pipelines for exploratory visualization of the data set:
1) A bar chart showing the distribution of labels in the training set
2) A visualization of one image for each unique label
3) A visualization of one image together with its label and text description of the label. The image can be adressed by its index in the training set

The pipelines make use of 
* numpys `unique()` function that returns the unique labels, the index to the first occurence of each unique label and the frequency (number of occurences) for each label
* pandas `read_csv()` function that is used to create a data frame from which the (numerical) labels can be converted to textual descriptions of the label using the `signnames.csv` file

Distirbution        | Label Examples
:------------------:|:-------------------------:
![alt text][image1] |  ![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it improves the efficiency when training and the relevant information for classifying the images is contained in the image itself.

As a second step, I decided to imrpove the contrast of the images by equalizing the histogram using opencvs function `equalizeHist()`
Here is an example of a traffic sign image before and after equalizing the histogram
![alt text][image9]

As last step, I normalized the image data using the following formula `(pixelvalue-128)/128` because the training is most efficient on pixel valus that have a distribution with mean=0 and std_deviation equal to both sides.

Here is an example of a traffic sign before and after preprocessing.
![alt_text][image10]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on LeNets architecture, but also includes dropouts. 
It is described in the following table:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2		| 2x2 stride, valid padding, outputs 14x14x6	|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling 2x2   	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Dropout			    | Dropout implemented							|
| Flattened 			| Flattened 5x5x16 to 1x400						|
| Fully connected		| Input 400, Outputs 120						|
| RELU					|												|
| Dropout			    | Dropout implemented							|
| Fully connected		| Input 120, Outputs 84							|
| RELU					|												|
| Dropout			    | Dropout implemented							|
| Fully connected		| Input 84, Outputs 43 (number of classes)		|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:
* EPOCHS = 50
* BATCH_SIZE = 128
* learning rate = 0.0005
* KEEP_PROB (for dropouts during training) = 0.8

I trained the model using the Adam Optimizer from `tf.train.AdamOptimizer`.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **99.9 %**
* validation set accuracy of **96.9 %**? 
* test set accuracy of **94.5 %**

The history of the validation accuracy during the training is visualized here:

![alt_tex][image12]

If a well known architecture was chosen:
* What architecture was chosen?\
I chose to use LeNets convolutional architecture as a basis and extended it with dropouts in the activation of the fully connected layers.
* Why did you believe it would be relevant to the traffic sign application?\
The LeNet was introduced in the corresponding lesson of the CarND and appeared to be useful for image classification tasks.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?\
All the accuracys are above 90%, so on these datasets the model is performing quite well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

| Image 1 | Image 2| Image 3 | Image 4 | Image 5 |
|:-------:|:------:|:-------:|:-------:|:-------:|
|![alt text][image4] | ![alt text][image5] | ![alt text][image6] | ![alt text][image7] | ![alt text][image8] |

The first image might be difficult to classify because it is covered with snow and includes a watermark in the middle of the picture.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ice/Snow (difficult)	| Right-of-way at next intersection				| 
| Ice/Snow (normal)		| Ice/Snow 										|
| End of all limits		| End of all limits								|
| Roundabout	      	| Keep right					 				|
| Right lane narrows	| Speed Limit (30 km/h)							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is way lower compared to the accuracy on the test set of 94.5%. The reason for this could lay in the different angles and positions of these web images that do not match the images from the training set. This could be reduced by augmenting the training set, i.e. resizing, cropping or tilting and by equalizing the number of samples from underrepresented labels.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The visualization below shows the probabilities of the top 5 softmax for each of the images from the web.
It can be seen that the prediction is quite certain (>80 % for each of the images), although 3 out of 5 images are classified incorrectly.

![alt_text][image11]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


