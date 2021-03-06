## Vehicle Detection and Tracking
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Writeup Pedro Marques
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/heat5.png
[image6]: ./output_images/mask_result.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook `vehicle_detection.ipynb`

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

There was no scientific method for choosing the HOG parameters, I simply tried multiple times with different settings until I found the one with the best result

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC using both the HOG and color features extracted  from the images, the car and not car features were put on a vertical stack then its mean and standard deviation where computed, and finally the features were standardized. The labels were organized as 1's for the car features and 0 for the not car features.

The next step was to split up this data (features and labels) into training and testing data, I randomly split it using 20 percent of the data for the testing set and remainder for training set. Having both sets I began training the classifier and then predicted the labels for the features on the test data.

The code for these steps are in the fifth cell of the IPython notebook `vehicle_detection.ipynb`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I had to do a lot of experimenting (really a lot) to decide the scales because of performance. To perform the sliding window search I had to specify which region I wanted to search first, how many pixels would I move my window on the x and y axis and how many windows I would have for that given region. The code for this implementation is on the sixth cell of the IPython notebook `vehicle_detection.ipynb`.

I have decided to use 64, 86 and 128 pixels sizes for my windows, I ran the 64 size through the entire search area, with no overlapping between them, I also had the 64 pixels again for the first part of the search area with 50% overlapping, I had the 86 and 128 pixels windows, both with 50% overlapping, run through the the first 128 pixels of the search area, again, I came up with these sizes through trying and failing. The code for this implementation is on the seventh cell of the IPython notebook `vehicle_detection.ipynb`.

Here is an image of the resulting windows:
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from one frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the frame of the video:

### Here is one frame and its corresponding heatmap:

![alt text][image5]

Another method I used was to find the outlines of the multiple boxes corresponding to car detections on a binary window, with the outlines I find the position and size, each of this outline will correspond to a vehicle on the image, with this data I can draw with confidence the box representing the car. Here is the result of this method:
![alt text][image6]

To smooth the box detection on the video I stored the last 20 frames on a queue and then I grouped those boxes together, needing at least 3 of them to retain.   
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Performance was definitely a concern here, when I was trying to implement the multiple sizes sliding windows I dived into a world of pain, I limited the search area and was using the scale variable on the `findcars` function, that I later switched to `findcars2` and forgot to remove the 2, I experimented with different scales but the results were not satisfactory.

I would detect when the vehicles are close and not when they got a little far and vice-versa, I thought, uhhmm eureka! I will use multiple scales and a little loop to go through all the scales, It began working much better but the time to process also got higher but the major problem was that the tracking was bad.

I got a great tip from my mentor to store a number of frames to a buffer and also group the detected boxes together and that was gold! I switched to the sliding method described before because of the time it was taking to process the video.
