
# coding: utf-8

# In[13]:

#import
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
get_ipython().magic('matplotlib notebook')


# In[14]:

def calibrate_camera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints,imgpoints


# In[15]:

def undistort_img(img, objpoints,imgpoints):
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )
    
    return dst


# In[16]:

def test_undistortion_img(input_img_path,output_img_path,objpoints, imgpoints):
    # Test undistortion on an image
    objpoints,imgpoints = calibrate_camera()
    img = cv2.imread(input_img_path)
    dst = undistort_img(img,objpoints,imgpoints)
    cv2.imwrite(output_img_path,dst)


# In[17]:

def main():
    objpoints,imgpoints = calibrate_camera()
    test_undistortion_img('camera_cal/calibration1.jpg','output_images/calibration1_undist.jpg',objpoints, imgpoints)

