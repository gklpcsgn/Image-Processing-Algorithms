import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Read the image

input = cv.imread('lena_grayscale_hq.jpg', cv.IMREAD_GRAYSCALE)

#########################################################
################# OPENCV INTEGRAL #######################
#########################################################

cv_integral = cv.integral(input)

#########################################################
#################### Question 1 #########################
#########################################################

integral_image = np.zeros(513*513).reshape(513,513)
for i in range(0,513):
    for j in range(0,513):
        integral_image[i][j] = np.sum(input[0:i,0:j])


#########################################################
#################### Question 2 #########################
#########################################################

def integral_box_filter(integral,kernel_size):
    
    padding_size = kernel_size//2

    integral_padded = np.pad(integral,(padding_size,0),'constant',constant_values=0)
    integral_padded = np.pad(integral_padded,(0,padding_size),'edge')

    normalize = kernel_size*kernel_size

    output = np.zeros((integral.shape[0]-1)*(integral.shape[1]-1)).reshape(integral.shape[0]-1,integral.shape[0]-1)
    for i in range(padding_size+1,integral_padded.shape[0]-padding_size):
        for j in range(padding_size+1,integral_padded.shape[0]-padding_size):
            output[i-(padding_size+1)][j-(padding_size+1)] = ((integral_padded[i+padding_size][j+padding_size] + integral_padded[i-padding_size-1][j-padding_size-1] - integral_padded[i-padding_size-1][j+padding_size] - integral_padded[i+padding_size][j-padding_size-1])/normalize + 0.5) //1
    
    output = output.astype(np.uint8)
    return output

integral_box = integral_box_filter(integral_image,3)


#########################################################
#################### Differences ########################
#########################################################

diff_1 = abs(integral_image - cv_integral)*100
diff_2 = abs(integral_box - cv.blur(input, (3,3), borderType=cv.BORDER_CONSTANT))

#########################################################
#################### Printing ###########################
#########################################################

print("Difference between OPENCV integral and MY integral: ", diff_1.sum())
print("Difference between OPENCV box filter and MY integral box filter: ", diff_2.sum())
