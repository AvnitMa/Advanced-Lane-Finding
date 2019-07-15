
# coding: utf-8

# In[1]:

import numpy as np
import prespective_transform as pt
import matplotlib.pyplot as plt
import cv2


# In[2]:

def get_histogram(binary_warped):
    # Take a histogram of the bottom half of the image
    values = int(binary_warped.shape[0]/2)
    histogram = np.sum(binary_warped[values:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    return histogram,midpoint


# In[3]:

def sliding_windows_line(binary_warped,base):
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
   # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    line_current = base
    
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    line_lane_inds = []
    
    
     # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xline_low = line_current - margin
        win_xline_high = line_current + margin
       
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xline_low,win_y_low),(win_xline_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_line_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xline_low) & (nonzerox < win_xline_high)).nonzero()[0]
        line_lane_inds.append(good_line_inds)
       
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_line_inds) > minpix:
            linex_current = np.int(np.mean(nonzerox[good_line_inds]))
            
     # Concatenate the arrays of indices
    line_lane_inds = np.concatenate(line_lane_inds)
    

    # Extract left and right line pixel positions
    linex = nonzerox[line_lane_inds]
    liney = nonzeroy[line_lane_inds] 
   
    line_fit = None
    line_fitx = None
    # Fit a second order polynomial to each
    if liney is not None and len(liney) > 0 and linex is not None and len(linex) > 0:
        line_fit = np.polyfit(liney, linex, 2)
   
     # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    if line_fit is not None:
        line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]
    
    return line_fit,line_fitx,ploty,line_lane_inds


# In[4]:

def sliding_windows_right(binary_warped):
    histogram,midpoint = get_histogram(binary_warped)
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    right_fit,right_fitx,ploty,right_lane_inds = sliding_windows_line(binary_warped,rightx_base)
    return right_fit,right_fitx,ploty,right_lane_inds


# In[5]:

def sliding_windows_left(binary_warped):
    histogram,midpoint = get_histogram(binary_warped)
    leftx_base = np.argmax(histogram[:midpoint])
    left_fit,left_fitx,ploty,left_lane_inds = sliding_windows_line(binary_warped,leftx_base)
    return left_fit,left_fitx,ploty,left_lane_inds


# In[6]:

def sliding_windows(binary_warped):
    left_fit,left_fitx,ploty,left_lane_inds = sliding_windows_left(binary_warped)
    right_fit,right_fitx,ploty,right_lane_inds = sliding_windows_right(binary_warped)
    return left_fit,right_fit,left_fitx,right_fitx,ploty,left_lane_inds,right_lane_inds


# In[7]:

def repeated_sliding_windows_line(binary_warped, line_fit):
    # Assume you now have a new warped binary image 
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    if len(line_fit)>1:
        line_lane_inds = ((nonzerox > (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] - margin)) & (nonzerox < (line_fit[0]*(nonzeroy**2) + line_fit[1]*nonzeroy + line_fit[2] + margin))) 
 
        # Again, extract left and right line pixel positions
        linex = nonzerox[line_lane_inds]
        liney = nonzeroy[line_lane_inds] 
 
        # Fit a second order polynomial to each
        if len(liney) > 0 and len(linex) > 0:
            line_fit = np.polyfit(liney, linex, 2)
   
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]

            return line_fit,line_fitx,ploty,line_lane_inds
        else:
            return None
    else:
        return None


# In[8]:

def repeated_sliding_windows(binary_warped,left_fit,right_fit):
    left_fit,left_fitx,ploty,left_lane_inds = repeated_sliding_windows_line(binary_warped, left_fit)
    right_fit,right_fitx,ploty,right_lane_inds = repeated_sliding_windows_line(binary_warped, right_fit)
    return left_fit,right_fit,left_fitx,right_fitx,ploty,left_lane_inds,right_lane_inds


# In[9]:

# Tests:


# In[10]:

def test_sliding_windows(binary_warped,left_lane_inds,right_lane_inds,left_fitx,right_fitx,ploty):
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig('output_images/test1_sliding_windows.png')


# In[11]:

def test_repeated_line_sliding_windows(binary_warped,left_lane_inds,right_lane_inds,left_fitx,right_fitx,ploty):
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig('output_images/test1_repeated_line_sliding_windows.png')


# In[12]:

def test_curvature(left_fitx,right_fitx,ploty):
    mark_size = 3
    plt.plot(left_fitx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(right_fitx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images
    plt.savefig('output_images/test1_curvature.png')


# In[13]:

def curv_line(ploty,fit):
    y_eval = np.max(ploty)
    line_curverad = None
    if fit is not None:
        line_curverad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return line_curverad


# In[14]:

# Define y-value where we want radius of curvature
def curv(ploty, left_fit, right_fit):
    y_eval = np.max(ploty)
    left_curverad = curv_line(ploty,left_fit)
    right_curverad = curv_line(ploty,right_fit)

    return left_curverad,right_curverad
# Example values: 1926.74 1908.48


# In[23]:

def curv_meters_line(ploty, line_fitx):
    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    line_curverad = None
    # Fit new polynomials to x,y in world space
    if line_fitx is not None:
        line_fit_cr = np.polyfit(ploty*ym_per_pix, line_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        line_curverad = ((1 + (2*line_fit_cr[0]*y_eval*ym_per_pix + line_fit_cr[1])**2)**1.5) / np.absolute(2*line_fit_cr[0])
        # Now our radius of curvature is in meters
    
    return line_curverad


# In[20]:

# Define conversions in x and y from pixels space to meters

def curv_meters(ploty, left_fitx, right_fitx):
    left_curverad = curv_meters_line(ploty, left_fitx)
    right_curverad = curv_meters_line(ploty, right_fitx)
    return left_curverad,right_curverad


# In[21]:

def main():
    image = cv2.imread('output_images/test1_undist.jpg')
    binary_warped, M, M_inv = pt.prespective_transform(image)

    left_fit,right_fit,left_fitx,right_fitx,ploty,left_lane_inds,right_lane_inds = sliding_windows(binary_warped)
    test_sliding_windows(binary_warped,left_lane_inds,right_lane_inds,left_fitx,right_fitx,ploty)
    left_fit,right_fit,left_fitx,right_fitx,ploty,left_lane_inds,right_lane_inds = repeated_sliding_windows(binary_warped,left_fit,right_fit)
    test_repeated_line_sliding_windows(binary_warped,left_lane_inds,right_lane_inds,left_fitx,right_fitx,ploty)
    test_curvature(left_fitx, right_fitx,ploty)
    print(curv(ploty, left_fit, right_fit))
    print(curv_meters(ploty, left_fitx, right_fitx))

