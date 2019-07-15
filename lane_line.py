
# coding: utf-8

# In[2]:

import numpy as np


# In[1]:

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,name):
        self.name = name
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.fits = []
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        self.points = None
        self.ploty = None
    def __str__(self):
        return ("Line name: "+self.name+" detected: "+
               str(self.detected)+ " recent_xfitted: "+
               str(self.recent_xfitted) +" bestx: "+
               str(self.bestx)+" best fit: "+ str(self.best_fit)+" current fit: "+
               str(self.current_fit)+" radius: "+str(self.radius_of_curvature)+
               " base pos: "+str(self.line_base_pos)+" diffs: "+str(self.diffs))
