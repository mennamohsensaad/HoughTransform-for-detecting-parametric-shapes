
# HoughTransform for detecting parametric shapes



## Objectives

* Apply Hough transform for detecting parametric shapes like circles and lines.
* Apply Harris operator for detecting corners.


### A) Computer Vision Functions

You need to implement Python functions which will support the following tasks:

1. For all given images; detect edges using Canny edge detector, detect lines and circles located in these images (if any). Superimpose the detected shapes on the images.

2. For given images; initialize the contour for a given object and evolve the Active Contour Model (snake) using the greedy algorithm. Represent the output as chain code and compute the perimeter and the area inside these contours.

You should implement these tasks **without depending on OpenCV library or alike**.


Add new Python files to organize implementation of the core functionalities:

1. `CV404Hough.py`: this will include your implementation for Hough transform for lines and circles (requirement 1).
2. `CV404Harris.py`: this will include your implementation for Harris operator for corners detection (requirement 2)

### B) GUI Integration

Integrate your functions in part (A) to the following Qt MainWindow design:



| Tab 4 |
|---|
| <img src=".screen/tab4.png" style="width:500px;"> |

| Tab 5 |
|---|
| <img src=".screen/tab5.png" style="width:500px;"> |
 
# Hough transform demo link :- https://drive.google.com/file/d/1ZR9RVGSdUJEo3qDwK-7gbJhUndcE1LTg/view?usp=drivesdk
 
# Harris demo link :- https://drive.google.com/file/d/1M3qzRWPmyPI4kx8lFsgfGbhWGSBxJSE2/view?usp=drivesdk
