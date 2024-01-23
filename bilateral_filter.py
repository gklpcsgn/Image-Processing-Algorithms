import numpy as np
import cv2 as cv

# Read the image

input_gaussian = cv.imread('noisyImage_Gaussian.jpg', cv.IMREAD_GRAYSCALE)
input_gaussian_0_1 = cv.imread('noisyImage_Gaussian_01.jpg',0) 
ground_truth = cv.imread('lena_grayscale_hq.jpg' , cv.IMREAD_GRAYSCALE)

#########################################################
################# OPENCV FILTERS  #######################
#########################################################

input_gaussian_normalized = cv.normalize(input_gaussian, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.single)

output_1_2 = cv.normalize(cv.blur(input_gaussian, (3,3), borderType = cv.BORDER_CONSTANT).astype(np.uint8), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)
output_1_3 = cv.normalize(cv.blur(input_gaussian, (5,5), borderType = cv.BORDER_CONSTANT).astype(np.uint8), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

output_1_4 = cv.normalize(cv.GaussianBlur(input_gaussian_normalized, (3,3),0, borderType = cv.BORDER_REPLICATE), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)
output_1_5 = cv.normalize(cv.GaussianBlur(input_gaussian_normalized, (5,5),0, borderType = cv.BORDER_REPLICATE), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

output_1_6 = cv.normalize(cv.bilateralFilter(input_gaussian_normalized,5, 3, 0.9, borderType = cv.BORDER_REPLICATE), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)


#########################################################
#################### Question 1 #########################
#########################################################

def adaptive_mean_filter(input,kernel_size,noise_var):

    padding_size = kernel_size//2

    output = np.zeros(input.shape)

    inp_padded = np.pad(input,(padding_size,padding_size),'edge')

    inp_padded = cv.normalize(inp_padded, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.single)
    
    for i in range(padding_size,inp_padded.shape[0]-padding_size):
        for j in range(padding_size,inp_padded.shape[0]-padding_size):
            values = inp_padded[i-padding_size : i+padding_size + 1, j-padding_size : j + padding_size + 1].flatten()
            local_var = np.var(values)
            local_mean = np.mean(values)
            output[i-padding_size,j-padding_size] = (inp_padded[i,j] - ((noise_var)/(local_var)) * (inp_padded[i,j] - local_mean))

    output = cv.normalize(output, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    output = (output + 0.5) // 1
    return output

output_1_1 = adaptive_mean_filter(input_gaussian,5,0.0042).astype(np.uint8)

#########################################################
#################### Question 2 #########################
#########################################################

input_gaussian_0_1_normalized = cv.normalize(input_gaussian_0_1, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.single)

output_2_2 = cv.normalize(cv.blur(input_gaussian_0_1_normalized, (3,3), borderType = cv.BORDER_REPLICATE), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)
output_2_3 = cv.normalize(cv.blur(input_gaussian_0_1_normalized, (5,5), borderType = cv.BORDER_REPLICATE), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

output_2_4 = cv.normalize(cv.GaussianBlur(input_gaussian_0_1_normalized, (3,3),0, borderType = cv.BORDER_REPLICATE), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)
output_2_5 = cv.normalize(cv.GaussianBlur(input_gaussian_0_1_normalized, (5,5),0, borderType = cv.BORDER_REPLICATE), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

output_2_6 = cv.normalize(cv.bilateralFilter(input_gaussian_0_1_normalized, 3, 0.1, 1, borderType = cv.BORDER_REPLICATE), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)

output_2_1 = adaptive_mean_filter(input_gaussian_0_1,5,0.0009).astype(np.uint8)

#########################################################
#################### Question 3 #########################
#########################################################

def bilateral_filter(input_image, sigma_r, sigma_s,kernel_size):

    input_image = cv.normalize(input_image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.single)

    output_image = np.zeros(input_image.shape)

    padding_size = int(kernel_size//2)
    
    input_image_padded = cv.copyMakeBorder(input_image, padding_size, padding_size, padding_size, padding_size, cv.BORDER_REPLICATE)
    input_image_padded = input_image_padded.astype(np.single)

    def gaussian(x, sigma):
        return np.exp(-1*x**2/(2*sigma**2)).astype(np.single)
    
    def distance(x,y, i,j):
        return np.sqrt((x-i)**2 + (y-j)**2)

    def distance_color(x,y):
        return abs(x-y).astype(np.single)

    def distance_color_gaussian(window,kernel_size, sigma_r):
        distance_color_gaussian = np.zeros((kernel_size, kernel_size))
        padding_size = int(kernel_size//2)
        # print("\nWindow: \n", window)
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance_color_gaussian[i,j] = gaussian(distance_color( window[padding_size, padding_size] , window[i,j] ), sigma_r)
        # print("\nDistance color gaussian: \n", distance_color_gaussian)
        return distance_color_gaussian


    distance_gaussian = np.zeros((kernel_size, kernel_size)).astype(np.single)
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance_gaussian[i,j] = gaussian(distance(i,j, padding_size, padding_size), sigma_s)
    # print("\nDistance gaussian: \n", distance_gaussian)


    for i in range(padding_size , input_image_padded.shape[0]-padding_size):
        for j in range(padding_size , input_image_padded.shape[1]-padding_size ):
            window = input_image_padded[i-padding_size:i+padding_size+1, j-padding_size:j+padding_size+1]
            color_matrix = distance_color_gaussian(window,kernel_size, sigma_r)
            # print("\nColor matrix: \n", color_matrix)
            weight_matrix = color_matrix * distance_gaussian
            # print("\nWeight matrix: \n", weight_matrix)
            weight_matrix = weight_matrix/np.sum(weight_matrix)
            output_image[i-padding_size,j-padding_size] = np.sum(window*weight_matrix)

    output_image = cv.normalize(output_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return output_image


#########################################################
#####################  Results  #########################
#########################################################

psnr_output_1_1 = str(cv.PSNR(output_1_1, ground_truth))
psnr_output_1_2 = str(cv.PSNR(output_1_2, ground_truth))
psnr_output_1_3 = str(cv.PSNR(output_1_3, ground_truth))
psnr_output_1_4 = str(cv.PSNR(output_1_4, ground_truth))
psnr_output_1_5 = str(cv.PSNR(output_1_5, ground_truth))
psnr_output_1_6 = str(cv.PSNR(output_1_6, ground_truth))

psnr_output_2_1 = str(cv.PSNR(output_2_1, ground_truth))
psnr_output_2_2 = str(cv.PSNR(output_2_2, ground_truth))
psnr_output_2_3 = str(cv.PSNR(output_2_3, ground_truth))
psnr_output_2_4 = str(cv.PSNR(output_2_4, ground_truth))
psnr_output_2_5 = str(cv.PSNR(output_2_5, ground_truth))
psnr_output_2_6 = str(cv.PSNR(output_2_6, ground_truth))



print(" ########## Question 1 ########## ")
print("PSNR of Gaussian Noisy Image: " + str(cv.PSNR(input_gaussian, ground_truth)))
print("PSNR of OPENCV Box Filter 3x3 : " + psnr_output_1_2)
print("PSNR of OPENCV Box Filter 5x5 : " + psnr_output_1_3)
print("PSNR of OPENCV Gaussian Filter 3x3 : " + psnr_output_1_4)
print("PSNR of OPENCV Gaussian Filter 5x5 : " + psnr_output_1_5)
print("PSNR of OPENCV Bilateral Filter : " + psnr_output_1_6)
print("PSNR of Adaptive Mean Filter : " + psnr_output_1_1)

print(" ########## Question 2 ########## ")
print("PSNR of Gaussian Noisy Image: " + str(cv.PSNR(input_gaussian_0_1, ground_truth)))
print("PSNR of OPENCV Box Filter 3x3 : " + psnr_output_2_2)
print("PSNR of OPENCV Box Filter 5x5 : " + psnr_output_2_3)
print("PSNR of OPENCV Gaussian Filter 3x3 : " + psnr_output_2_4)
print("PSNR of OPENCV Gaussian Filter 5x5 : " + psnr_output_2_5)
print("PSNR of OPENCV Bilateral Filter : " + psnr_output_2_6)
print("PSNR of Adaptive Mean Filter : " + psnr_output_2_1)

print(" ########## Question 3 ########## ")

input_gaussian_01 = cv.imread('noisyImage_Gaussian_01.jpg',0) 

output_q3 = bilateral_filter(input_gaussian, 3,0.9, 5)

gaussian_normalized_q3 = cv.normalize(input_gaussian, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.single)
output_cv_q3 = cv.bilateralFilter(gaussian_normalized_q3, 5, 3, 0.9, cv.BORDER_REPLICATE)
output_cv_q3 = cv.normalize(output_cv_q3, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

diff = abs(output_q3 - output_cv_q3)

diff[diff<10] = 0

print(np.max(abs(output_q3 - output_cv_q3)))

output_01_q3 = bilateral_filter(input_gaussian_01, 1,0.1, 3)

gaussian_normalized_01_q3 = cv.normalize(input_gaussian_01, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.single)
output_cv_01_q3 = cv.bilateralFilter(gaussian_normalized_01_q3, 3, 1, 0.1)
output_cv_01_q3 = cv.normalize(output_cv_01_q3, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

diff_01 = abs(output_01_q3 - output_cv_01_q3)

print(np.max(abs(output_01_q3 - output_cv_01_q3)))

psnr_output_3_1 = str(cv.PSNR(output_q3.astype(np.uint8), ground_truth))
psnr_output_3_2 = str(cv.PSNR(output_01_q3.astype(np.uint8), ground_truth))

print("PSNR of Gaussian Noisy Image: " + str(cv.PSNR(input_gaussian, ground_truth)))
print("PSNR of My Bilateral Filter: " + psnr_output_3_1)
print("PSNR of My Bilateral Filter for 01: " + psnr_output_3_2)


print("Max difference between my bilateral filter and opencv bilateral filter: " + format(np.max(diff), '.2f'))
print("Max difference between my bilateral filter and opencv bilateral filter for 01: " + format(np.max(diff_01), '.2f'))


cv.imshow('Gaussian Noisy Image : ' + str(cv.PSNR(input_gaussian, ground_truth)), input_gaussian)
cv.imshow('OPENCV Box Filter 3x3 : ' + psnr_output_1_2, output_1_2)
cv.imshow('OPENCV Box Filter 5x5 : ' + psnr_output_1_3, output_1_3)
cv.imshow('OPENCV Gaussian Filter 3x3 : ' + psnr_output_1_4, output_1_4)
cv.imshow('OPENCV Gaussian Filter 5x5 : ' + psnr_output_1_5, output_1_5)
cv.imshow('OPENCV Bilateral Filter : ' + psnr_output_1_6, output_1_6)
cv.imshow('Adaptive Mean Filter : ' + psnr_output_1_1, output_1_1)

cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow('Gaussian Noisy Image : ' + str(cv.PSNR(input_gaussian_0_1, ground_truth)), input_gaussian_0_1)
cv.imshow('OPENCV Box Filter 3x3 : ' + psnr_output_2_2, output_2_2)
cv.imshow('OPENCV Box Filter 5x5 : ' + psnr_output_2_3, output_2_3)
cv.imshow('OPENCV Gaussian Filter 3x3 : ' + psnr_output_2_4, output_2_4)
cv.imshow('OPENCV Gaussian Filter 5x5 : ' + psnr_output_2_5, output_2_5)
cv.imshow('OPENCV Bilateral Filter : ' + psnr_output_2_6, output_2_6)
cv.imshow('Adaptive Mean Filter : ' + psnr_output_2_1, output_2_1)

cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow('Gaussian Noisy Image : ' + str(cv.PSNR(input_gaussian, ground_truth)), input_gaussian)
cv.imshow('My Bilateral Filter : ' + psnr_output_3_1, output_q3.astype(np.uint8))
cv.imshow('OPENCV Bilateral Filter : ' + psnr_output_1_6, output_1_6)
cv.imshow("Difference : " + format(np.max(diff), '.2f'), diff)

cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow('Gaussian Noisy Image : ' + str(cv.PSNR(input_gaussian_0_1, ground_truth)), input_gaussian_0_1)
cv.imshow('My Bilateral Filter : ' + psnr_output_3_2, output_01_q3.astype(np.uint8))
cv.imshow('OPENCV Bilateral Filter : ' + psnr_output_2_6, output_2_6)
cv.imshow("Difference : " + format(np.max(diff_01), '.2f'), diff_01)

cv.waitKey(0)
cv.destroyAllWindows()



