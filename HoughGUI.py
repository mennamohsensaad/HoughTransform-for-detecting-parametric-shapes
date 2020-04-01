
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
import handleImageClass as handleImage





    
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
        self.ui.comboBox_shape.currentIndexChanged.connect(self.Hough_space)
        self.ui.pushButton_Hough_load.clicked.connect(self.LoadImage)
        self.ui.Hough_ApplyButton.clicked.connect(self.Hough_space) #self.Apply_canny
        self.ui.pushButton_opencv.clicked.connect(self.Apply_canny_with_opencv)
        #img =handleImage.imarray(object)
        
        

      
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
            #print(pixels.shape)
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
#    def gaussian(self,m,n,sigma):
#        gaussian=np.zeros((m,n))
#        m=m//2
#        n=n//2
#        for x in range (-m,m+1):
#            for y in range (-n,n+1):
#                x1=sigma*math.sqrt(2*np.pi)
#                x2=np.exp(-(x**2+y**2)/(2*sigma**2))
#                gaussian[x+m,y+n]=(1/x1)*x2  
#        return gaussian
    

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
#___________________________________to check accuracy of our implementation to canny edges detector ____________________
#___________________________  try canny with opencv to compare results with our implemetation for canny_________________
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

#________________________________________________HoughFunctions___________________________#
       
       
    

#________________________________________________________gaussian & sobel & laplacian __________________________________#
    
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
    
    def gaussian_filter(self,m,n,sigma,img):
        g=self.gaussian(m,n,sigma)
        img.convolve(g)
        return img
        
    
    def edge(self,img,threshold):
        # Laplacian with sobel to detect the edges
        laplacian = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        print(sum(sum(laplacian)))
        sobel_x = ([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = ([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        gx = img.convolve(sobel_x)
        gy = img.convolve(sobel_y)
    
        G = np.sqrt(gx ** 2 + gy ** 2) 
        
    
        G[G<threshold] = 0
        Lap = img.convolve(laplacian)
    
        M,N = Lap.shape
    
        temp = np.zeros((M+2,N+2))                                                 
        temp[1:-1,1:-1] = Lap                                                       
        result = np.zeros((M,N))                                                   
        for i in range(1,M+1):
            for j in range(1,N+1):
                #Looking for a negative pixel and checking its 8 neighbors
                if temp[i,j]<0:                                                    
                    for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                            if temp[i+x,j+y]>0:
                                result[i-1,j-1] = 1                                
        #img.load(np.array(np.logical_and(result,G),dtype=np.uint8))
        img.load(np.array(np.logical_and(result,G),dtype=np.uint8))
        return img
    
    def detectCircles(self,img,threshold,region,radius):
        M,N = img.shape
        [R_max,R_min] = radius
    
        R = R_max - R_min
        #3D accumulator array :radius, X ,Y 
        A = np.zeros((R_max,M+2*R_max,N+2*R_max))
        B = np.zeros((R_max,M+2*R_max,N+2*R_max))
    
        theta = np.arange(0,360)*np.pi/180
        #edges = np.nonzero(img[:,:])
        #For non Zero elements
        edges = np.argwhere(img[:,:])
                                                      
        for K in range(R):
            r = R_min+K
            
            circle = np.zeros((2*(r+1),2*(r+1)))
            #Finding out the center 
            (m,n) = (r+1,r+1)                                                       
            for angle in theta:
                x = int(np.round(r*np.cos(angle)))
                y = int(np.round(r*np.sin(angle)))
                circle[m+x,n+y] = 1
            constant = np.argwhere(circle).shape[0]
            #print(circle)
            #print(constant)
            for x,y in edges:                                                     
                X = [x-m+R_max,x+m+R_max]                                           
                Y= [y-n+R_max,y+n+R_max]                                            
                A[r,X[0]:X[1],Y[0]:Y[1]] += circle
            A[r][A[r]<threshold*constant/r] = 0
    
        for r,x,y in np.argwhere(A):
            temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
            try:
                p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
            except:
                continue
            B[r+(p-region),x+(a-region),y+(b-region)] = 1
    
        return B[:,R_max:-R_max,R_max:-R_max]   
           
           
    
    def displayCircles(self,A):
        img = imread(self.fileName)
        fig =plt.figure()
        plt.imshow(img)
        circleCoordinates = np.argwhere(A)                                          
        circle = []
        for r,x,y in circleCoordinates:
            circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
            fig.add_subplot(111).add_artist(circle[-1])
            
        #self.ui.label_filters_input.setPixmap(self.input_iamge)
        #plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
#        plt.savefig("fff.jpg")
       
        
        plt.savefig('fff.jpg', dpi=900, bbox_inches='tight',pad_inches=0)
        
        
        
        
    def Hough_space(self):
        self.hough = str(self.ui.comboBox_shape.currentText())
        print(self.hough)
        if self.hough=="Lines":
            self.filter_img =self.im_gaussian_noise(0,0.3)
        elif self.hough=="Circles": 
            input_size=(self.ui.lineEdit_mask_size.text())
            print(input_size)
            #self.fileName
            self.img =handleImage.imarray(self.fileName)
            self.filter_img =self.gaussian_filter(int(input_size),int(input_size),2,self.img)            
            self.image = self.edge(self.filter_img,128) 
            
                              
            self.image = self.detectCircles(self.image,13,15,[40,10])
            self.output_iamge=self.displayCircles(self.image)
#            elif self.fileName=="images10.jpg":       
#                 self.image = self.detectCircles(self.image,9,15,[40,10])
#                 self.output_iamge=self.displayCircles(self.image)
           
            #self.output_iamge=self.displayCircles(self.image)
            
                  
            
            #self.ui.label_Hough_output.setPixmap(self.output_iamge)
            self.ui.label_Hough_output.setPixmap(QPixmap("fff.jpg"))
            self.ui.label_Hough_output.show()
        
        
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    application =Hough()
    application.show()

    
  
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()



