**Advanced Lane Finding Project**

In this project:

* I Computed the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Applied a distortion correction to raw images.
* Used color transforms, gradients, etc., to create a thresholded binary image.
* Applied a perspective transform to rectify binary image ("birds-eye view").
* Detected lane pixels and fit to find the lane boundary.
* Determined the curvature of the lane and vehicle position with respect to center.
* Warped the detected lane boundaries back onto the original image.
* Outputed visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

##### Camera Calibration

###### 1. Computation of the camera matrix and distortion coefficients

The code for this step is contained in lines # through # of the file called `camera_caliberation_and_correcting_distortion.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted](http://i.imgur.com/9ogO9zN.png)
##### Pipeline (single images)

###### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Test1](http://i.imgur.com/F9Lsfab.jpg)

###### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in `thresholded_binary_images.py`).  Here's an example of my output for this step.  

![Thresholded image](http://i.imgur.com/9D53yxJ.png)

I also tried other methods for generate a binary image like sobel, a red channel, and HLS color extraction for the yellow and white lane lines.

![Yellow white lines](http://i.imgur.com/QYg1Ua9.png)
![Sobel](http://i.imgur.com/eA6MSjt.png)
![Red](http://i.imgur.com/iL6RPmJ.png)

###### 3. Perspective transform the image

The code for my perspective transform includes a function called `prespective_transform()`, which appears in lines 1 through 8 in the file `prespective_transform.py`.  The `prespective_transform()` function takes as inputs an image (`img`), and generate source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Prespective Transform](http://i.imgur.com/A9lCumj.png)

###### 4. Identification of the lane-line pixels

I fit my lane lines with a 2nd order polynomial in sliding_windows_and_curvature.py
As described in the lesson, I first took a histogram of the warped 
threshold to identify the base of the lines. Then, I used the Sliding Windows technique to idetentify the lane lines.

![Histogram](http://i.imgur.com/6eR65jR.png)
![Sliding Windows](http://i.imgur.com/p2WHyNT.png)
![Repeat Sliding Windows](http://i.imgur.com/DD2WETh.png)


###### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center

I did this in lines #223 through #282 in my code in `sliding_windows_and_curvature.py`

###### 6. An example image of the result plotted back down onto the road

I implemented this step in `advance_lines_detection` notebook in the function `process_images(image)`. Here is an example of my result on a test image:

![Result plotted](http://i.imgur.com/HUYMGkZ.jpg)

---

##### Pipeline (video)

[![Output](http://i.imgur.com/CaH1AnQ.png)](https://vimeo.com/229826472 "Output")


---

##### Discussion

I saw that in some conditions like curvy lines and darker images , the lane lines are not being detected, so I decided to make the image brighter and do a sanity on the lines so I will be able to use previous good data to draw the lines.
However, the pipeline might fail if there are different pavement colors on the same road. One solution might be to analyze the image diffrently (maybe use a different cylindrical-coordinate representation of the RGB color model).

