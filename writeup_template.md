
## **Vehicle Detection Project**


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
[image8]: ./writeup_images/hog1.png
[image9]: ./writeup_images/hog2.png
[image10]: ./writeup_images/test1.png
[image11]: ./writeup_images/test2.png
[image12]: ./writeup_images/test3.png
[image13]: ./writeup_images/test4.png
[image14]: ./writeup_images/test5.png
[image15]: ./writeup_images/test6.png
[image16]: ./writeup_images/labels1.png

##### Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/513/view) points individually and describe how I addressed each point in my implementation. Most of the code is from the course material itself. I understood the code and used it when appropriate.

---

#### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cell ...

Using `skimage.hog()` we can extract the hog features, as follows:

```python
orientations = 9
pixels_per_cell = 8
cell_per_block = 2
features, hog_image1 = get_hog_features(gray1, orientations,
pixels_per_cell, cell_per_block, vis=True, feature_vec=False)
```
Next, I visualized  two random car and notcar images along with their corresponding hog features.

![alt text][image8]

![alt text][image9]

Using HOG features, one can easily spot the differences between the car and non-car images. This might be difficult in case of color histogram and spacial bins.

#### 2. Explain how you settled on your final choice of HOG parameters.

The final parameters I used for this project are

* colorspace = 'YCrCb'
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL'
* hist_bins = 32
* spatial_size = (32, 32)

The above choice was made from following some discussions in December Facebook group, slack group, and also after trying different colorspaces manually. Ideally, I have to use grid search on validation data and pick the parameters that that give rise to least validation error. However, I think the current choice of parameters are good enough since they give an accuracy of 0.99 on the test set.



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Refer code in cell block 10, 11, and 12.

The following code is used for splitting the data into training and test sets.
```python
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
```

The training is done as follows:
```python
# Use a linear SVC
svc = LinearSVC()
svc.fit(X_train, y_train)
```
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the code (from the course material) in cell 7 the function ``find_cars`` uses the scale parameter [0.75, 1.0, 2.0, 3.0] and the search region increases with increase in scale parameter [400, 500], [400, 550], [400, 600] , [400, 700]. My choice of using scaled implementation over sliding window was it implicitly takes care of the bounding box sizes and instead of overlap the code defines number of cells to step.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The following are results from the pipeline on the test images:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

A threshold of 2 was used to achieve the above images.
The feature vectors used consists of hog features, histogram features and spacial features.


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The code in cell 16 (``find_cars_final``) results in the final video.

Here's a [link to my video result](./project_video_lanes_final.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code in cell 13 uses the threshold on the final heatmap to remove false positives. From the final heatmap using label from (``from scipy.ndimage.measurements import label ``) results in blobs, as shown below

![alt text][image16]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* I did find few false positives. Furthermore, the choice of scale parameter over fixed-size sliding windows will can result in weird small boxes on the vehicles instead of a clear bounding box.

Improvements

* Implementing a YOLO convolution neural network that predicts the bounding boxes could be one way to go.

* Tracking the centroid of the blob could also result in smooth tracking.
