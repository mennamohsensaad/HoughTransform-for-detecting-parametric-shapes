import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import gaussian
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.util import img_as_float
from skimage.filters import sobel
import cv2 as cv 




def normalize(array, newMin, newMax):   #to normalize from 0 to 1
    minArr=np.min(array)
    maxArr=np.max(array)
    return ((array-minArr)/(maxArr-minArr))*(newMax-newMin)+newMin
 



def interpolation(x_list,y_list):
    new_x = []
    new_y = []
    lenth = len(x_list)   #length of contour 
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



def active_contour_model(imageName,x_list,y_list):
    img =io.imread(imageName,0)
    color_img = img
    gray_img = rgb2gray(img)
    init_contour = np.array([x_list, y_list]).T
    #print(init_contour)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.plot(init_contour[:, 0], init_contour[:, 1], '-r', lw=3)
    ax.set_title('Initial contour')
    gaussian_img =gaussian(gray_img,2)
    #gaussian_img_norm=normaliz(gaussian_img, minVal=0, maxVal=1)
    snake = active_contour( gaussian_img,color_img,
                           init_contour, alpha=0.015, beta=10, gamma=0.001)
    

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title('Final contour')
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()
    return snake


def active_contour(image,color, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   bc='periodic', max_px_move=1.0,
                   max_iterations=1000, convergence=0.25):

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
            #plt.imshow(sobel(img), cmap=plt.get_cmap('gray'))
        for i in range(3 if RGB else 1):
            edge[i][0, :] = edge[i][1, :]
            edge[i][-1, :] = edge[i][-2, :]
            edge[i][:, 0] = edge[i][:, 1]
            edge[i][:, -1] = edge[i][:, -2]
    else:
        edge = [0]
     
#    img_norm= #normalize(edge,0,1)
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
#    fig, ax = plt.subplots(figsize=(7, 7))
#    ax.imshow(color)


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
        """show iteration of running snake open the following : """
        """fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.get_cmap('gray'))
        #ax.plot(np.array([x, y]), '-b', lw=3)       
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        ax.set_title('iteration : %s'%(i))
        ax.set_xticks([]), ax.set_yticks([])
        plt.show(block=False), plt.pause(0.001)
        del ax.lines[0]
    plt.close()
    """
    return np.array([x, y]).T
#____________________________________________chain_code____________________________________________________________

"""
we use a hash function. The difference in X(dx) and Y(dy) co-ordinates of two successive points are calculated and hashed to generate the key for the chain code between the two points.

Chain code list: [5, 6, 7, 4, 0, 3, 2, 1]

Hash function:  C(dx, dy) = 3dy + dx + 4
"""


codeList = [5, 6, 7, 4, 0, 3, 2, 1] 
  
  
# This function generates the chaincode  
# for transition between two neighbour points 
def getChainCode(x1, y1, x2, y2): 
    dx = x2 - x1 
    dy = y2 - y1 
    hashKey = 3 * dy + dx + 4
   
    if  int(hashKey) >7:
             hashKey=7
    elif int(hashKey) <0 :
             hashKey=0
    else:
            hashKey=hashKey
    #print(hashKey )
    return codeList[int(hashKey)] 
  
#'''This function generates the list of  
#chaincodes for given list of points'''
def generateChainCode(ListOfPoints): 
    chainCode = [] 
    for i in range(len(ListOfPoints) - 1): 
        a = ListOfPoints[i] 
        b = ListOfPoints[i + 1] 
        chainCode.append(getChainCode(a[0], a[1], b[0], b[1])) 
    return chainCode 

def DriverFunction(ListOfPoints): 

    chainCode = generateChainCode(ListOfPoints)
    #print(chainCode)
    chainCodeString = "".join(str(e) for e in chainCode)
    print ('Chain code for the contour is', chainCodeString) 
           
    return chainCode



def calc_area_preimeter(snake):
        
        area=0
        primeter=0
        for i in range(len(snake)-1): 
            a =snake[i] 
            b=snake[i + 1] 
            #print(b)
            x,y=a[0],a[1]
            nex_x,nex_y=b[0],b[1]
            area += (x*nex_y-y*nex_x)/10
            primeter +=abs((x-nex_x)+(y-nex_y)*1j)/5
        area = area / 12
        return area,primeter


imageName = "./images/cell.jpeg"
r=100 #radius
c=[100,100] #center
theta=np.linspace(0, 2*np.pi, 200) # min, max, number of divisions
x=c[0]+r*np.cos(theta)
y=c[1]+r*np.sin(theta)
x_list = x
#print(x_list)
y_list = y
img = cv2.imread(imageName,0)
x_list,y_list=interpolation(x_list,y_list)
list_s=active_contour_model(imageName,x_list,y_list)
chainCode=DriverFunction(list_s)     
area,perimeter=calc_area_preimeter(list_s)
print("area of contour is :",area )
print("perimeter of contour is :",perimeter)


