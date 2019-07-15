
# coding: utf-8

# In[14]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
import thresholded_binary_images as th
get_ipython().magic('matplotlib inline')


# In[15]:

def set_variables(img_size):

    src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) )-10, img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

    return src,dst


# In[20]:

def prespective_transform(img):
    img_size = (img.shape[1], img.shape[0])
    src,dst = set_variables(img_size)
    thresholded = th.select_yellow_white(img)
    img_size = (thresholded.shape[1], thresholded.shape[0])
    
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(thresholded, M, img_size , flags=cv2.INTER_LINEAR)
    
    return warped, M, M_inv


# In[21]:

def test_prespective_transform(image):
    copy = image.copy()
    vrx = np.array(([203, 720],[1127, 720],[695, 460],[585, 460]), np.int32)
    vrx = vrx.reshape((-1,1,2))
    img = cv2.polylines(copy, [vrx], True, (255,0,0),3)
    warped, M, M_inv = prespective_transform(img)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(copy)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped, cmap='gray')
    ax2.set_title('Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('output_images/test1_warped.png')


# In[22]:

def main():
    image = cv2.imread('output_images/test1_undist.jpg')
    test_prespective_transform(image)


