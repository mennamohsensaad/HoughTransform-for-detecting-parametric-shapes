

####@Nashwa

import numpy as np
from  qimage2ndarray import array2qimage
from scipy import ndimage
import matplotlib.image as mpimg
from scipy.ndimage.filters import convolve
from PIL import Image
from numpy import asarray
import math
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import imread
import handleImageClass as handleImage
from collections import defaultdict
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
#hough transform

def Hough_Line(image):
    #######First find image dimensions
    # y for rows and x for columns 
    Y = image.shape[0]
    X = image.shape[1]
    DIAGONAL = int(np.round(np.sqrt(X**2 + Y ** 2)))
    thetas = np.deg2rad(np.arange(0, 180))
    ########what is radius Range
    rs = np.linspace(-DIAGONAL, DIAGONAL, 2*DIAGONAL)
    #2. Create accumulator array and initialize to zero
    accumulator = np.zeros((2 * DIAGONAL, len(thetas)))
    #3. Loop for each edge pixel 
    for y in range(Y):
        for x in range(X):
            # Check if it is an edge pixel
            #  NB: y -> rows , x -> columns
            if image[y,x] > 0:
                #4. Loop for each theta
                # Map edge pixel to hough space
                 for k in range(len(thetas)):

                    #5. calculate $\rho$
                    # Calculate space parameter
                    r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])

                    accumulator[int(r) + DIAGONAL,k] += 1
    return accumulator, thetas, rs
#extract lines from image
#######Draw__lines
def Draw_Lines(accumulator,thetas,rhos,threshold):
                  lines = defaultdict()
                  
                  acc2 = np.zeros(accumulator.shape)
                  for rho_idx in range(len(rhos)) :
                    for theta_idx in range(len(thetas)) :
                        if accumulator[rho_idx, theta_idx] > threshold :
                            theta = thetas[theta_idx]
                            rho = rhos[rho_idx]
                            # print (angle , rho , accumulator[angle_index , rho])
                            lines[(rho,theta)] = accumulator[rho_idx, theta_idx]
                            
                            acc2[rho_idx,theta_idx] = accumulator[rho_idx, theta_idx]
                  return lines,acc2
              
def IMPOSE_Lines ():
              
    img = cv2.imread('images/Chess_Board.svg.png') 
    
    #img = cv2.imread('images/beauflor-spirit-chessboard-vinyl-flooring-p755-3264_image.jpg')
    # Convert the img to grayscale 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    
    # Apply edge detection method on the image 
    edges = cv2.Canny(gray,50,150,apertureSize = 3) 
    #cv2.imshow('gray',gray)
    #cv2.imshow('edges',edges)
    
    #test hough line 
    accumulator,thetas,rhos = Hough_Line(edges)
    
    lines, acc2 = Draw_Lines( accumulator,thetas,rhos,90)
    #cv2.imshow('acc2',acc2)
    
    
    for (rho,theta), val in lines.items():
       
        a = np.cos(theta)
        b = np.sin(theta)
        pt0 = rho * np.array([a,b])
        pt1 = tuple((pt0 + 1000 * np.array([-b,a])).astype(int))
        pt2 = tuple((pt0 - 1000 * np.array([-b,a])).astype(int))
        cv2.line(img, pt1, pt2, (0,0,255), 3)
    
        cv2.line(img, pt1, pt2, (0,0,255), 3)
    #cv2.imshow('image with lines',img) 	
    #
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #  
    plt.imshow(img)
    #    plt.subplot(122), plt.imshow(accumulator)
    plt.show()
IMPOSE_Lines ()