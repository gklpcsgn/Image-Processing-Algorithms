import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Read the image

input = cv.imread('noisyImage.jpg', cv.IMREAD_GRAYSCALE)
ground_truth = cv.imread('lena_grayscale_hq.jpg', cv.IMREAD_GRAYSCALE) 

#########################################################
################# OPENCV MEDIAN FILTER ##################
#########################################################

output_cv_median = cv.medianBlur(input,5, cv.BORDER_REPLICATE)

#########################################################
#################### Question 1 #########################
#########################################################

def median_filter(img,kernel_size):

    #Apply padding to the image
    padding_size = kernel_size // 2
    img_padded = np.pad(input, (padding_size, padding_size), 'edge')

    #Create output image
    output = np.zeros(img.shape)

    #Apply median filter
    for i in range(padding_size,img_padded.shape[0]-padding_size):
        for j in range(padding_size,img_padded.shape[0]-padding_size):
            values = img_padded[i-padding_size:i+padding_size+1, j-padding_size:j+padding_size+1].flatten()

            median = np.median(values)
            output[i-padding_size,j-padding_size] = median
    output = (output + 0.5) // 1
    return output

output_my_median = median_filter(input,5)

#########################################################
#################### Question 2 #########################
#########################################################

output_cv_box = cv.blur(input, (5,5), borderType = cv.BORDER_REPLICATE)
output_cv_gauss = cv.GaussianBlur(input, (5, 5), 0, borderType = cv.BORDER_REPLICATE)

#########################################################
#################### Question 3 #########################
#########################################################

def weighted_median_filter(img,kernel_size):

    #Apply padding to the image
    padding_size = kernel_size // 2
    img_padded = np.pad(input, (padding_size, padding_size), 'edge')
    
    center_weight = 3
    
    #Create output image
    output = np.zeros(img.shape)

    #Apply median filter
    for i in range(padding_size,img_padded.shape[0]-padding_size):
        for j in range(padding_size,img_padded.shape[0]-padding_size):
            values = img_padded[i-padding_size:i+padding_size+1, j-padding_size:j+padding_size+1].flatten()
            values = np.append(values, [img_padded[i,j]] * center_weight)

            median = np.median(values)
            output[i-padding_size,j-padding_size] = median
    output = (output + 0.5) // 1
    return output


output_my_weighted_median = weighted_median_filter(input,5)
#########################################################
#################### Question 4 #########################
#########################################################

#  We just add a black border to the image, it significantly affecting the psnr value but human eye can not see the difference

best_psnr = output_my_weighted_median.copy()
best_psnr[0:1,0:512] = 0
best_psnr[0:512,0:1] = 0
best_psnr[511:512,0:512] = 0
best_psnr[0:512,511:512] = 0



#########################################################
#################### Differences ########################
#########################################################

diff_1_1 = abs(output_cv_median - output_my_median)

#########################################################
#################### Printing ###########################
#########################################################

psnr_output_cv_box = str(cv.PSNR(output_cv_box, ground_truth, 255))
psnr_output_cv_gauss = str(cv.PSNR(output_cv_gauss, ground_truth, 255))
psnr_output_cv_median = str(cv.PSNR(output_cv_median, ground_truth, 255))
psnr_output_my_median = str(cv.PSNR(output_my_median.astype(np.uint8), ground_truth, 255))
psnr_output_my_weighted_median = str(cv.PSNR(output_my_weighted_median.astype(np.uint8), ground_truth, 255))
psnr_best_psnr = str(cv.PSNR(best_psnr.astype(np.uint8), ground_truth, 255))

print("Difference between OPENCV median filter and MY median filter: ", diff_1_1.sum())
print("PSNR value for OPENCV box filter: ", psnr_output_cv_box)
print("PSNR value for OPENCV gauss filter: ", psnr_output_cv_gauss)
print("PSNR value for OPENCV median filter: ", psnr_output_cv_median)
print("PSNR value for MY median filter: ", psnr_output_my_median)
print("PSNR value for MY weighted median filter: ", psnr_output_my_weighted_median)
print("PSNR value for low psnr good image : ", psnr_best_psnr)

#########################################################
#################### Results ############################
#########################################################

#display image with opencv

cv.imshow('output_cv_box : ' + psnr_output_cv_box,  output_cv_box.astype(np.uint8))
cv.imshow('output_cv_gauss : ' + psnr_output_cv_gauss,  output_cv_gauss.astype(np.uint8))
cv.imshow('output_cv_median : ' + psnr_output_cv_median,  output_cv_median.astype(np.uint8))
cv.imshow('output_my_median : ' + psnr_output_my_median,  output_my_median.astype(np.uint8))
cv.imshow('output_my_weighted_median : ' + psnr_output_my_weighted_median,  output_my_weighted_median.astype(np.uint8))
cv.imshow('low_psnr_good_image : ' + psnr_best_psnr,  best_psnr.astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()