from PyQt5 import QtWidgets,QtGui , QtCore ,Qt
from PyQt5.QtWidgets import   QFileDialog  ,QWidget,QApplication
from PyQt5.QtGui import QPixmap
from MainWindow import Ui_MainWindow
from PIL import Image
from PIL.ImageQt import ImageQt
import sys
from os import listdir
from os.path import isfile , join
import numpy as np
from  qimage2ndarray import array2qimage
from scipy import ndimage
#import array2qimage 
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.filters import convolve
from numpy import asarray
import cv2


class Hough(QtWidgets.QMainWindow):
    def __init__(self):
        super(Hough, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.imgs_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = 75
        self.strong_pixel = 255
        self.sigma = 1
        self.kernel_size = 5
        self.lowThreshold = 0.05
        self.highThreshold = 0.15
        self.ui.pushButton_Hough_load.clicked.connect(self.LoadImage)
        self.ui.Hough_ApplyButton.clicked.connect(self.Apply_canny)
        self.ui.pushButton_opencv.clicked.connect(self.Apply_canny_with_opencv)

      
    def LoadImage(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if self.fileName:
            pixmap = QPixmap(self.fileName)
            self.pixmap = pixmap.scaled(256,256, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation) 
            self.input_img =mpimg.imread(self.fileName)
            #self.gray =cv2.imread(self.fileName,0)
            self.gray_img =self.rgb2gray( self.input_img)
            #plt.imshow(self.color_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
            self.ui.label_Hough_input.setPixmap(self.pixmap)
            self.ui.label_Hough_input.show
            #to show size of the image 
            pixels = asarray(self.input_img)
            print(pixels.shape)
            self.ui.lineEdit_size_Hough.setText(""+str(pixels.shape[0])+" "+str('x')+" "+str(pixels.shape[1])+"")
    
    def rgb2gray(self,rgb):
        #return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])
          r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
          gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
          return gray 


    
    def Apply_canny(self):
       self.canny_filter()
       self.ui.label_Hough_output.setPixmap(QPixmap("canny_edges.jpg"))
    
#    
    def canny_filter(self):
              #self.gray_img
#              
            #self.gray_img =cv2.imread(self.fileName,0)
            canny_img_final = []
            input_size=(self.ui.lineEdit_mask_size.text())
            #plt.imshow(self.gray_img)
            guass=self.gaussian(int(input_size),int(input_size),1)
            self.img_smoothed = convolve(self.gray_img,guass)
            self.gradientMat, self.thetaMat =self.sobel_filter_for_canny(self.img_smoothed)
            self.nonMaxImg =self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg =self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            canny_img_final.append(img_final)
            print(canny_img_final)
            self.visualize(canny_img_final, 'gray')                 
    def gaussian(self,m,n,sigma):
        gaussian=np.zeros((m,n))
        m=m//2
        n=n//2
        for x in range (-m,m+1):
            for y in range (-n,n+1):
                x1=sigma*math.sqrt(2*np.pi)
                x2=np.exp(-(x**2+y**2)/(2*sigma**2))
                gaussian[x+m,y+n]=(1/x1)*x2  
        return gaussian
    

    def sobel_filter_for_canny(self,img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    
    def non_max_suppression(self,img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z
    
    def threshold(self,img):
#        lowThreshold = 0.05
#        self.highThreshold = 0.1
        LhighThreshold = img.max() * self.highThreshold;
        LlowThreshold = self.highThreshold * self.lowThreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= LhighThreshold)
        zeros_i, zeros_j = np.where(img < LlowThreshold)

        weak_i, weak_j = np.where((img <= LhighThreshold) & (img >= LlowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self,img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img


    def visualize(self,imgs, format=None, gray=False):
            #plt.figure(figsize=(20, 40))
            for i, img in enumerate(imgs):
                if img.shape[0] == 3:
                    img = img.transpose(1,2,0)
                img=Image.fromarray(np.uint8(img))    
                img.save("canny_edges.jpg")    

    def Apply_canny_with_opencv(self):    
       img = cv2.imread(self.fileName,0)
       edges = cv2.Canny(img,100,200)
       print (edges)
       pixels=np.array(edges)
        #gray2qimage
       iamge=array2qimage(pixels)
       pixmap = QPixmap( iamge)
       self.ui.label_Hough_output_2.setPixmap(pixmap)
       self.ui.label_Hough_output_2.show



def main():
    app = QtWidgets.QApplication(sys.argv)
    application =Hough()
    application.show()
  
    
  
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


