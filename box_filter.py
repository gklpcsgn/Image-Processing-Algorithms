import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Read the image

input = cv.imread('lena_grayscale_hq.jpg', cv.IMREAD_GRAYSCALE)

#########################################################
################# OPENCV BOX FILTER #####################
#########################################################

output_2_1 = cv.blur(input, (3,3), borderType=cv.BORDER_CONSTANT)
output_2_2 = cv.blur(input, (11,11), borderType=cv.BORDER_CONSTANT)
output_2_3 = cv.blur(input, (21,21), borderType=cv.BORDER_CONSTANT)

#########################################################
#################### Question 1 #########################
#########################################################

def box_filter(img, kernel_size):

    # Create a kernel with all ones
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    
    #Apply padding to the image
    padding_size = kernel_size // 2
    img_padded = np.pad(input, (padding_size, padding_size), 'constant', constant_values=(0, 0))
    
    #Create normalize constant
    normalize = kernel_size * kernel_size
    
    #Create output image
    output = np.zeros(img.shape)

    #Apply box filter
    for i in range(padding_size,img_padded.shape[0]-padding_size):
        for j in range(padding_size,img_padded.shape[0]-padding_size):
            output[i-padding_size,j-padding_size] = ((img_padded[i-padding_size:i+padding_size+1, j-padding_size:j+padding_size+1] * kernel).sum()/normalize + 0.5) // 1
    return output

output_1_1 = box_filter(input, 3)
output_1_2 = box_filter(input, 11)
output_1_3 = box_filter(input, 21)

#########################################################
#################### Question 2 #########################
#########################################################

def seperable_filter(img, kernel_size):

    # Create horizontal and vertical kernels with all ones
    kernel_v = np.ones((1,kernel_size), np.float32)
    kernel_h = kernel_v.T

    #Apply padding to the image
    padding_size = kernel_size // 2
    img_padded = np.pad(input, (padding_size, padding_size), 'constant', constant_values=(0, 0))
    
    #Create normalize constant
    output = np.zeros(img.shape)

    #Apply horizontal filter
    for i in range(padding_size,img_padded.shape[0]-padding_size):
        for j in range(padding_size,img_padded.shape[0]-padding_size):
            output[i-padding_size,j-padding_size] = ((img_padded[i,j-padding_size:j+padding_size+1].reshape(kernel_size,1)*kernel_h).sum()/kernel_size)

    #add padding to output
    output_first = np.pad(output, (padding_size, padding_size), 'constant', constant_values=(0, 0))
    
    #Apply vertical filter
    for i in range(padding_size,output_first.shape[0]-padding_size):
        for j in range(padding_size,output_first.shape[0]-padding_size):
            output[i-padding_size,j-padding_size] = ((output_first[i-padding_size:i+padding_size+1,j]*kernel_v).sum()/kernel_size + 0.5) // 1
    return output

output_3_1 = seperable_filter(input, 3)
output_3_2 = seperable_filter(input, 11)
output_3_3 = seperable_filter(input, 21)

#########################################################
#################### Differences ########################
#########################################################

diff_1_1 = abs(output_1_1 - output_2_1)
diff_1_2 = abs(output_1_2 - output_2_2)
diff_1_3 = abs(output_1_3 - output_2_3)

diff_2_1 = abs(output_3_1 - output_2_1)
diff_2_2 = abs(output_3_2 - output_2_2)
diff_2_3 = abs(output_3_3 - output_2_3)


#########################################################
#################### Printing ###########################
#########################################################

print("Difference between output_1_1 and output_2_1: ", diff_1_1.sum())
print("Difference between output_1_2 and output_2_2: ", diff_1_2.sum())
print("Difference between output_1_3 and output_2_3: ", diff_1_3.sum())

print("Difference between output_3_1 and output_2_1: ", diff_2_1.sum())
print("Difference between output_3_2 and output_2_2: ", diff_2_2.sum())
print("Difference between output_3_3 and output_2_3: ", diff_2_3.sum())

#########################################################
#################### Results ############################
#########################################################

#display image with opencv

cv.imshow('output_1_1', output_1_1.astype(np.uint8))
cv.imshow('output_1_2', output_1_2.astype(np.uint8))
cv.imshow('output_1_3', output_1_3.astype(np.uint8))

cv.imshow('output_3_1', output_3_1.astype(np.uint8))
cv.imshow('output_3_2', output_3_2.astype(np.uint8))
cv.imshow('output_3_3', output_3_3.astype(np.uint8))

cv.imshow('diff_1_1', diff_1_1.astype(np.uint8))
cv.imshow('diff_1_2', diff_1_2.astype(np.uint8))
cv.imshow('diff_1_3', diff_1_3.astype(np.uint8))


cv.imshow('diff_2_1', diff_2_1.astype(np.uint8))
cv.imshow('diff_2_2', diff_2_2.astype(np.uint8))
cv.imshow('diff_2_3', diff_2_3.astype(np.uint8))


cv.waitKey(0)
cv.destroyAllWindows()