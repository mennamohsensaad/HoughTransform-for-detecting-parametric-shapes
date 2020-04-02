# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:35:24 2020

@author: شيماء
"""

from PyQt5 import QtWidgets,QtGui , QtCore ,Qt
from PyQt5.QtWidgets import   QFileDialog  ,QWidget,QApplication
from PyQt5.QtGui import QPixmap
from MainWindow import Ui_MainWindow
import numpy as np
from  qimage2ndarray import array2qimage
from scipy import ndimage
import matplotlib.image as mpimg
from scipy.ndimage.filters import convolve
from PIL import Image
from numpy import asarray
import cv2
import math
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import imageio
from scipy import ndimage



class Harris(QtWidgets.QMainWindow):
        def __init__(self):
            super(Harris, self).__init__()
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)
            self.imgs_final = []
    
            self.ui.pushButton_Harris_load.clicked.connect(self.LoadImage)
    #        self.ui.Hough_ApplyButton.clicked.connect(self.Hough_space) #self.Apply_canny
            #self.ui.Harris_ApplyButton_2.clicked.connect(self.Apply_canny_with_opencv)
            #img =handleImage.imarray(object)
            
        
        
        def LoadImage(self):  
            self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG *.JFIF);;img file (*.PNG)")
            if self.fileName:
                pixmap = QPixmap(self.fileName)
                self.pixmap = pixmap.scaled(256,256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation) 
                self.input_img =mpimg.imread(self.fileName)
                #self.gray =cv2.imread(self.fileName,0)
                self.gray_img =self.rgb2gray( self.input_img)
                #plt.imshow(self.color_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
                self.ui.label_Harris_input.setPixmap(self.pixmap)
                self.ui.label_Harris_input.show
                #to show size of the image 
                pixels = asarray(self.input_img)
                #print(pixels.shape)
                self.ui.lineEdit_size_Hough.setText(""+str(pixels.shape[0])+" "+str('x')+" "+str(pixels.shape[1])+"")    
        
        #Define RGB2gray function
        def rgb2gray(self,img) :
             return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
         
            
        #Detemine gradient function for Fx and Fy using sobel filter(normlized)
        def gradient_x(self) :
            grad_img = ndimage.convolve(self.gray_img, np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]))
            return grad_img/np.max(grad_img)
        
        def gradient_y(self) :
            grad_img = ndimage.convolve(self.gray_img, np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]))
            return grad_img/np.max(grad_img)
        
        
        def harris(self):
            ratio = [0.05, 0.05, 0.05, 0.05, 0.01, 0.001]
            count = 0 #for equivalent ratio access
            #Apply gaussian blurring
            blur_img = ndimage.gaussian_filter( self.gray_img, sigma = 1.0)
            #Find gradient Fx
            x_grad = self.gradient_x(blur_img)
            
            #Find gradient Fy
            y_grad = self.gradient_y(blur_img)
            
            
                
            #Phase II : Find corners
            xx_grad = x_grad * x_grad
            yy_grad = y_grad * y_grad
            xy_grad = x_grad * y_grad
            tuple_data = [] #Contains y, x Co-ordinates and its corner response
            k = 0.08
            max = 0
            
            for i in range(1, int(self.fileName.shape[0] - 1)) :
                for j in range(1, int(self.fileName.shape[1] - 1)) :
                    window_x = xx_grad[i-4 : i+5 , j-4 : j+5]
                    window_y = yy_grad[i-4 : i+5 , j-4 : j+5]
                    window_xy = xy_grad[i-4 : i+5 , j-4 : j+5] 
                    sum_xx = np.sum(window_x)
                    sum_yy = np.sum(window_y)
                    sum_xy = np.sum(window_xy)
                    determinant = (sum_xx * sum_yy) - (sum_xy * sum_xy)
                    trace = sum_xx + sum_yy
                    R = determinant - (k * trace * trace)
                    tuple_data.append((i, j, R))
                    if(R > max) :
                        max = R
        
                    #L contains y, x co-ordinate(whose value is greater than threshold) and their corner response of those co-ordinates
                    L = []
                    thres_ratio = ratio[count]
                    count+=1
                    threshold = thres_ratio * max
                    for res in tuple_data :
                        i, j, R = res
                        if R > threshold :
                            L.append([i, j, R])
                          
                    
                    
                    #Phase III : Non maximal suppression
                    sorted_L = sorted(L, key = lambda x: x[2], reverse = True)
                    final_L = [] #final_l contains list after non maximal suppression
                    final_L.append(sorted_L[0][:-1])
                    dis = 15
                    self.xc, self.yc = [], []
                    for i in sorted_L :
                        for j in final_L :
                            if(abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis) :
                                break
                        else :
                            final_L.append(i[:-1])
                            self.xc.append(i[1])
                            self.yc.append(i[0])
                    
        

	    
            
            
            
        
        def Apply(self):
                #self.click=0
                self.pixmap = QPixmap(self.gray_img)
                self.ui.label_snake_input.setPixmap(self.pixmap)
                plt.imshow(self.fileName, cmap = plt.get_cmap('gray'))
                plt.plot(self.xc, self.yc, '*', color='purple')
                
                self.ui.label_Harris_output.self.harris()
                self.ui.label_Harris_output.show() 
                #plt.imshow(self.pixmap)
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    application =Harris()
    application.show()

    
  
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()        