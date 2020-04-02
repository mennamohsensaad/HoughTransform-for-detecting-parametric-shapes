# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:17:43 2020

@author: شيماء
"""

from pylab import *
from scipy import *
from scipy import signal
from PIL import Image

def gaussian_derivative_kernels(size, sizey=None):
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = mgrid[-size:size+1, -sizey:sizey+1]

    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    gy = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 

    return gx,gy

def gauss_derivatives(im, n, ny=None):

    gx,gy = gaussian_derivative_kernels(n, sizey=ny)

    Ix = signal.convolve(im,gx, mode='same')
    Iy = signal.convolve(im,gy, mode='same')

    return Ix,Iy

def compute_harris_response(image):

    #derivatives
    Ix,Iy = gauss_derivatives(image, 3)

    #kernel for blurring
    gaussian = gauss_kernel(3)
    
    #compute components of the structure tensor
    Ixx = signal.convolve(Ix*Ix,gaussian, mode='same')
    Ixy = signal.convolve(Ix*Iy,gaussian, mode='same')
    Iyy = signal.convolve(Iy*Iy,gaussian, mode='same')
    
    #determinant and trace
    Idet = Ixx*Iyy - Ixy**2
    Itr = Ixx + Iyy
    
    return Idet / Itr

def get_harris_points(harrisim, min_distance=10, threshold=0.1):
  
    #find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    #get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]
    
    #sort candidates
    index = argsort(candidate_values)
    
    #store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1
    
    #select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),
                (coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0
                
    return filtered_coords

def gauss_kernel(size, sizey = None):
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def plot_harris_points(image, filtered_coords):
 
        
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
    axis('off')
    show()


img = array(Image.open('./images/girlWithScarf.png').convert("L"))
harrisim =compute_harris_response(img)
filtered_coords = get_harris_points(harrisim,6)
plot_harris_points(img, filtered_coords)
    

