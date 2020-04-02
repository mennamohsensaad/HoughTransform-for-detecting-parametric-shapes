from PyQt5 import QtWidgets,QtGui , QtCore ,Qt
from PyQt5.QtWidgets import   QFileDialog  ,QWidget,QApplication
from PyQt5.QtGui import QPixmap,QPainter,QPen
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
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import gaussian
from skimage.filters import sobel
from scipy.interpolate import RectBivariateSpline
from skimage.util import img_as_float
from scipy.ndimage import imread
import handleImageClass as handleImage
from collections import defaultdict






class CV(QtWidgets.QMainWindow):
    def __init__(self):
        super(CV, self).__init__()
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
        self.click=0
        #self.ui.comboBox_shape.currentIndexChanged.connect(self.IMPOSE_Lines_Circles)
        self.ui.pushButton_Hough_load.clicked.connect(self.LoadImage)
        self.ui.pushButton_apply_hough.clicked.connect(self.IMPOSE_Lines_Circles)
        self.ui.pushButton_opencv.clicked.connect(self.Apply_canny_with_opencv)
        self.ui.pushButton_snake_load.clicked.connect(self.LoadImage2)
        self.ui.Reset.clicked.connect(self.Reset)
        self.ui.clear_anchors.clicked.connect(self.Clear_anchors)
        self.ui.snake_apply.clicked.connect(self.Apply_snake)
        self.ui.label_snake_input.mousePressEvent=self.getPixel
        self.ui.pushButton_Harris_load.clicked.connect(self.LoadImage3)
    
    def LoadImage(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if self.fileName:
            self.pixmap = QPixmap(self.fileName)
            self.input_img =mpimg.imread(self.fileName)
            self.ui.label_Hough_input.setPixmap(self.pixmap)
            self.ui.label_Hough_output.clear()
            self.ui.label_Hough_output_2.clear()
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
            #check if gray image or color 
            pixels = asarray(self.input_img)
            pixels = pixels.astype('float32')
            try:
                if (pixels.shape[2] == 3 or pixels.shape[2] == 4 ):
                    self.gray_img =self.rgb2gray( self.input_img)
                   
            #elif (pixels.shape[2] == 1):
            except IndexError:  #(pixels.shape[2] == None):
                     self.gray_img =self.input_img
                     
        
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
       iamge=array2qimage(pixels*1000)
       pixmap = QPixmap( iamge)
       self.ui.label_Hough_output_2.setPixmap(pixmap)
       self.ui.label_Hough_output_2.show
       
       
#________________________________________________HoughFunctions__________________________________________________________
       
       
    

#           ________________________________gaussian & sobel & laplacian __________________
    
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
       
        
        plt.savefig('detection_of_circles.jpg', dpi=900, bbox_inches='tight',pad_inches=0)
        
        
        
        



#####################################################HOUGH-TRAANSFORM-LINE-DETECTION###############333

#hough transform
    def Hough_Line(self,image):
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
    def Draw_Lines(self,accumulator,thetas,rhos,threshold):
                  self.lines = defaultdict()
                  
                  self.acc2 = np.zeros(accumulator.shape)
                  for rho_idx in range(len(rhos)) :
                    for theta_idx in range(len(thetas)) :
                        if accumulator[rho_idx, theta_idx] > threshold :
                            theta = thetas[theta_idx]
                            rho = rhos[rho_idx]
                            # print (angle , rho , accumulator[angle_index , rho])
                            self.lines[(rho,theta)] = accumulator[rho_idx, theta_idx]
                            
                            self.acc2[rho_idx,theta_idx] = accumulator[rho_idx, theta_idx]
                  return self.lines,self.acc2
              

              
    def IMPOSE_Lines_Circles (self):
     
     self.hough = str(self.ui.comboBox_shape.currentText())
       
     print(self.hough)
     if self.hough=="Circles":
            self.ui.label_Hough_output.clear()
            self.ui.label_Hough_output_2.clear() 
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
            self.ui.label_Hough_output.setPixmap(QPixmap("detection_of_circles.jpg"))
            self.ui.label_Hough_output.show()
            
     elif self.hough=="Lines": 
        self.ui.label_Hough_output.clear()
        self.ui.label_Hough_output_2.clear() 
        input_size=(self.ui.lineEdit_mask_size.text())
        print(input_size)      
        imag = cv2.imread(self.fileName) 
        
        #img = cv2.imread('images/beauflor-spirit-chessboard-vinyl-flooring-p755-3264_image.jpg')
        # Convert the img to grayscale 
        gray = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY) 
        # Apply edge detection method on the image 
        self.edges = cv2.Canny(gray,50,150,apertureSize = 3) 
        #cv2.imshow('gray',gray)
        #cv2.imshow('edges',edges)
        #test hough line 
        accumulator,thetas,rhos = self.Hough_Line(self.edges)
        self.lines, self.acc2 = self.Draw_Lines( accumulator,thetas,rhos,90)
        #cv2.imshow('acc2',acc2)
        
        for (rho,theta), val in self.lines.items():
           
            a = np.cos(theta)
            b = np.sin(theta)
            pt0 = rho * np.array([a,b])
            pt1 = tuple((pt0 + 1000 * np.array([-b,a])).astype(int))
            pt2 = tuple((pt0 - 1000 * np.array([-b,a])).astype(int))
            cv2.line(imag, pt1, pt2, (0,0,255), 3)
        
            cv2.line(imag, pt1, pt2, (0,0,255), 3)   
        pixels=np.array(imag)
        #gray2qimage
        im=array2qimage(pixels)
        pixmap = QPixmap(im)
        self.ui.label_Hough_output.setPixmap(pixmap)
        #self.ui.label_Hough_output.show
        #plt.imshow(pixmap)
        
    
    
     elif self.hough =="Canny" : 
           self.ui.label_Hough_output.clear()
           self.canny_filter()
           self.ui.label_Hough_output.setPixmap(QPixmap("canny_edges.jpg"))   
          
     else :  

          self.ui.label_Hough_output.clear()       
  



#____________________________________________Active_contour_Model_________________________________________________
  
    def LoadImage2(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if self.fileName:
            self.Reset()
            self.pixmap = QPixmap(self.fileName)
            self.input_img =mpimg.imread(self.fileName)
            self.ui.label_snake_input.setPixmap(self.pixmap)
            self.ui.label_snake_output.clear()
            self.ui.label_snake_output_running.clear()
            pixels = asarray(self.input_img)
            #print(pixels.shape)
            self.ui.lineEdit_size_snake.setText(""+str(pixels.shape[0])+" "+str('x')+" "+str(pixels.shape[1])+"")
            
    
    def Apply_snake(self):
        
        snake_list=self.active_contour_model(self.fileName)
        #print(snake_list)
        chainCode=self.DriverFunction(snake_list) 
        self.ui.lineEdit_chain_code.setText(""+str(chainCode)+"")#+str('x')+""+str(" ")+"")
        
    def intp(self,x_list,y_list):
        new_x = []
        new_y = []
        lenth = len(x_list)
        print('contour lenghth : ',lenth)
        if lenth < 400:
            for i in range(1,lenth-1):
                new_x.append(x_list[i - 1])
                new_y.append(y_list[i - 1])
                inp_x = (x_list[i-1] + x_list[i]) //2
                inp_y = (y_list[i-1] + y_list[i]) //2
    
                new_x.append(inp_x)
                new_y.append(inp_y)
                #print(new_x,new_y)        
        else:
            return x_list,y_list
        return new_x,new_y
    
    
    def active_contour_model(self,imageName):
        img =io.imread(imageName,0)
        color_img = img
        gray_img = rgb2gray(img)
        gaussian_img =gaussian(gray_img,2)
        #gaussian_img_norm=normalizeRange(gaussian_img, minVal=0, maxVal=1)
        #signal.convolve2d(gray_img, gaussian_Filter(5, (3,3)), mode='same') #gaussian(gray_img,3)  #convolve image with guassian fi
        alpha=float(self.ui.lineEdit_alpha.text())
        beta=float(self.ui.lineEdit_beta.text())
        gamma=float(self.ui.lineEdit_gamma.text())
        snake = self.active_contour( gaussian_img,color_img,
                               self.init_contour, alpha, beta, gamma)
        " alpha=0.015, beta=10, gamma=0.001"
        QApplication.processEvents()
        self.pixmap = QPixmap(self.fileName)
        painter= QtGui.QPainter(self.pixmap)
        painter.begin(self)
        penRect= QtGui.QPen(QtCore.Qt.green)
        penRect.setWidth(4)
        painter.setPen(penRect)
        for i in range(len(snake) - 1): 
            a = snake[i] 
            painter.drawPoint(a[0],a[1])
        painter.end()
        result=self.pixmap#.scaled(int(self.pixmap.height()),int(self.pixmap.width()))
        self.ui.label_snake_output.setPixmap(result)
        QApplication.processEvents()
        return snake
    
    def inital_contour(self,):
        c=[self.x,self.y]
        r=np.sqrt((self.x-self.x1) ** 2+(self.y-self.y1) ** 2)
        theta=np.linspace(0, 2*np.pi, 200) # min, max, number of divisions
        x=c[0]+r*np.cos(theta)
        y=c[1]+r*np.sin(theta)
        x_list = x
        #print ( x_list)
        y_list = y
        new_x_list,new_y_list=self.intp(x_list,y_list)
        self.init_contour = np.array([new_x_list,new_y_list]).T
        #print(init)
        QApplication.processEvents()
        painter= QtGui.QPainter(self.pixmap)
        painter.begin(self)
        penRect= QtGui.QPen(QtCore.Qt.red)
        penRect.setWidth(4)
        painter.setPen(penRect)
        for i in range(len(self.init_contour)): 
            a = self.init_contour[i] 
            #print(b)
            painter.drawPoint(a[0],a[1])
        
        painter.end()
        result=self.pixmap#.scaled(int(self.pixmap.height()),int(self.pixmap.width()))
        self.ui.label_snake_input.setPixmap(result)
        QApplication.processEvents()
    
    def active_contour(self,image,color, snake, alpha, beta,gamma,
                   w_line=0, w_edge=1, 
                   bc='periodic', max_px_move=1.0,
                   max_iterations=3000, convergence=0.25):
        max_iterations = int(max_iterations)
        if max_iterations <= 0:
            raise ValueError("max_iterations should be >0.")
        convergence_order = 10
        img = img_as_float(image)
        
        RGB = img.ndim == 3
    
        if w_edge != 0:
            if RGB:
                edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]),
                        sobel(img[:, :, 2])]
            else:
                edge = [sobel(img)]
                
    
            for i in range(3 if RGB else 1):
                edge[i][0, :] = edge[i][1, :]
                edge[i][-1, :] = edge[i][-2, :]
                edge[i][:, 0] = edge[i][:, 1]
                edge[i][:, -1] = edge[i][:, -2]
        else:
            edge = [0]
            
        #print(edge)    
    #    img_norm=normalizeRange(edge, minVal=0, maxVal=1)  #normalize(edge,0,1)
    #    edge=img_norm
        if RGB:
            img = w_line*np.sum(img, axis=2) \
                + w_edge*sum(edge)
        else:
            img = w_line*img + w_edge*edge[0]
    
    
        intp = RectBivariateSpline(np.arange(image.shape[1]), #img
                                   np.arange(image.shape[0]),
                                   img.T, kx=2, ky=2, s=0)
    
    
        x, y = snake[:, 0].astype(np.float), snake[:, 1].astype(np.float)
        xsave = np.empty((convergence_order, len(x)))
        ysave = np.empty((convergence_order, len(x)))
    
    
        n = len(x)
    
        a = np.roll(np.eye(n), -1, axis=0) + np.roll(np.eye(n), -1, axis=1) - 2*np.eye(n)
        b = np.roll(np.eye(n), -2, axis=0) + \
            np.roll(np.eye(n), -2, axis=1) - \
            4*np.roll(np.eye(n), -1, axis=0) - \
            4*np.roll(np.eye(n), -1, axis=1) + \
            6*np.eye(n)
        A = -alpha*a + beta*b
        #print(A.shape)
        #print((gamma*np.eye(n)).shape)
    
        inv = np.linalg.inv(A + gamma*np.eye(n))
#        fig, ax = plt.subplots(figsize=(7, 7))
#        ax.imshow(color)
    
    
        for i in range(max_iterations):
    
            fx = intp(x, y, dx=1, grid=False)
            fy = intp(x, y, dy=1, grid=False)
    
            xn = inv @ (gamma*x + fx)
            yn = inv @ (gamma*y + fy)
    
    
            dx = max_px_move*np.tanh(xn-x)
            dy = max_px_move*np.tanh(yn-y)
    
            x += dx
            y += dy
    
            j = i % (convergence_order+1)
            if j < convergence_order:
                xsave[j, :] = x
                ysave[j, :] = y
            else:
                dist = np.min(np.max(np.abs(xsave-x[None, :]) +  np.abs(ysave-y[None, :]), 1))
                if dist < convergence:
                    break
    
            
            snake = np.array([x, y]).T
              
            #pause time for some sec befor next iteration 
            sec=0
            self.ui.lineEdit_iteration_num.setText(""+str(i)+"")
            for i in range(9000):
                sec=sec+1   
            
            self.running_snake(img,snake)
 
        return np.array([x, y]).T
    
    def running_snake(self,img,snake):
        QApplication.processEvents()
        #self.pixmap = QPixmap(img)
        pixels=np.array(img)
         #gray2qimage
        iamge=array2qimage(pixels*1000)
        #new_img = Image.fromarray(iamge)
        pixmap = QPixmap(iamge)
        #self.pixmap = QPixmap(self.fileName)
        painter= QtGui.QPainter(pixmap)
        painter.begin(self)
        penRect= QtGui.QPen(QtCore.Qt.blue)
        penRect.setWidth(4)
        painter.setPen(penRect)
        for i in range(len(snake)): 
            a =  snake[i] 
            #print(b)
            painter.drawPoint(a[0],a[1])
        
        painter.end()
        result=pixmap#.scaled(int(self.pixmap.height()),int(self.pixmap.width()))
        self.ui.label_snake_output_running.setPixmap(result)
        QApplication.processEvents()
        
    def getPixel(self,event):
        
          if (self.click==0):
              self.x=math.floor((event.pos().x()*self.pixmap.width())/self.ui.label_snake_input.frameGeometry().width()) 
              self.y=math.floor((event.pos().y()*self.pixmap.height())/self.ui.label_snake_input.frameGeometry().height())
              self.click=self.click+1  
              print(self.x,self.y)
            
          elif(self.click==1):
              self.x1=math.floor((event.pos().x()*self.pixmap.width())/self.ui.label_snake_input.frameGeometry().width()) 
              self.y1=math.floor((event.pos().y()*self.pixmap.height())/self.ui.label_snake_input.frameGeometry().height())
              self.click=self.click+1
              print(self.x1,self.y1)
          self.frame_on_Anchors()
              
    def frame_on_Anchors(self):
        
        QApplication.processEvents()
        painter= QtGui.QPainter(self.pixmap)
        painter.begin(self)
        penRect= QtGui.QPen(QtCore.Qt.red)
        penRect.setWidth(2)
        painter.setPen(penRect)
        if (self.click==1):
           painter.drawRect(self.x,self.y,4, 4)
        elif (self.click==2):
           painter.drawRect(self.x1,self.y1,4, 4)
           
        else:
            print("end")
        painter.end()
        result=self.pixmap#.scaled(int(self.pixmap.height()),int(self.pixmap.width()))
        self.ui.label_snake_input.setPixmap(result)
        QApplication.processEvents()
        if (self.click==2):
             self.inital_contour()
        else:
            self.click=self.click
    

    def Clear_anchors(self):
        self.click=0
        self.pixmap = QPixmap(self.fileName)
        self.ui.label_snake_input.setPixmap(self.pixmap)
        self.ui.label_snake_output.clear()
        self.ui.label_snake_output_running.clear() 
        
    
    def Reset(self): 
        self.Clear_anchors()
#        pixels=np.ones((10,10))
#        iamge=array2qimage(pixels*1000)
#        pixmap = QPixmap(iamge)
#        self.ui.label_snake_output.setPixmap(pixmap)
#        self.ui.label_snake_output_running.clear() 
        self.ui.lineEdit_chain_code.setText(""+str("")+"")
        self.ui.lineEdit_iteration_num.setText(""+str("")+"")
        self.ui.lineEdit_alpha.setText(""+str("")+"")
        self.ui.lineEdit_beta.setText(""+str("")+"")
        self.ui.lineEdit_gamma.setText(""+str("")+"")
        
        
    def DriverFunction(self,ListOfPoints): 
        chainCode = self.generateChainCode(ListOfPoints)
        #print(chainCode)
        chainCodeString = "".join(str(e) for e in chainCode)
        #print (chainCodeString)
        return chainCodeString
    
    
    def getChainCode(self,x1, y1, x2, y2): 
        codeList = [5, 6, 7, 4, -1, 0, 3, 2, 1]     # to detect direction
        dx = x2 - x1 
        dy = y2 - y1 
        hashKey = 3 * dy + dx + 4
        return codeList[int(hashKey)] 

    def generateChainCode(self,ListOfPoints): 
        chainCode = [] 
        for i in range(len(ListOfPoints) - 1): 
            a = ListOfPoints[i] 
    #        print(a)
            #print(a[0])
            b = ListOfPoints[i + 1] 
            #print(b)
            chainCode.append(self.getChainCode(a[0], a[1], b[0], b[1])) 
        return chainCode 
 
    
#_________________________________________Harris_corner_detectors____________________________________________________    
     
    def LoadImage3(self):  
        self.fileName, _filter = QFileDialog.getOpenFileName(self, "Title"," " , "Filter -- img file (*.jpg *.PNG);;img file (*.PNG)")
        if self.fileName:
            self.pixmap = QPixmap(self.fileName)
            self.input_img =mpimg.imread(self.fileName)
            self.ui.label_Harris_input.setPixmap(self.pixmap)
            self.ui.label_Harris_output.clear()
            pixels = asarray(self.input_img)
            #print(pixels.shape)
            self.ui.lineEdit_size_Harris.setText(""+str(pixels.shape[0])+" "+str('x')+" "+str(pixels.shape[1])+"")
def main():
    app = QtWidgets.QApplication(sys.argv)
    application =CV()
    application.show()
  
    
  
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

