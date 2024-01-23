import numpy as np
import cv2 as cv

# Read the image

input_gaussian = cv.imread('noisyImage_Gaussian.jpg', cv.IMREAD_GRAYSCALE)
input_salt_pepper = cv.imread('noisyImage_SaltPepper.jpg', cv.IMREAD_GRAYSCALE)
ground_truth = cv.imread('lena_grayscale_hq.jpg', cv.IMREAD_GRAYSCALE)

#########################################################
################# OPENCV FILTERS  #######################
#########################################################

output_1_2 = cv.blur(input_gaussian, (5,5), borderType = cv.BORDER_CONSTANT).astype(np.uint8)
output_1_3 = cv.GaussianBlur(input_gaussian, (5,5),0, borderType = cv.BORDER_CONSTANT).astype(np.uint8)

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

output_1_1 = adaptive_mean_filter(input_gaussian,5,0.004).astype(np.uint8)

#########################################################
#################### Question 2 #########################
#########################################################

def adaptive_median_filter(input):

    def Level_B(z_min,z,z_max,z_med):
        if z_min < z < z_max:
            return z
        else:
            return z_med

    def Level_A(S,S_max,image,i,j):
        padding_size = S//2
        values = image[i-padding_size : i+padding_size + 1, j-padding_size : j + padding_size + 1].flatten()

        z_min = np.min(values)
        z_max = np.max(values)
        z_med = np.median(values)
        z = inp_padded[i,j]

        if z_min < z_med < z_max:
            return Level_B(z_min,z,z_max,z_med)
        else:
            if S == S_max:
                return np.median(values)
            return Level_A(S+2,S_max,image,i,j)

    S = 3
    S_max = 7
    padding_size_S_max = S_max//2
    output = np.zeros(input.shape)

    inp_padded = np.pad(input,(padding_size_S_max,padding_size_S_max),'edge')
    inp_padded = cv.normalize(inp_padded, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.float64)

    for i in range(padding_size_S_max,inp_padded.shape[0]-padding_size_S_max):
        for j in range(padding_size_S_max,inp_padded.shape[0]-padding_size_S_max):
            output[i-padding_size_S_max,j-padding_size_S_max] = Level_A(S,S_max,inp_padded,i,j)

    output = cv.normalize(output, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    output = (output + 0.5) // 1

    return output

output_2_1 = adaptive_median_filter(input_salt_pepper).astype(np.uint8)
output_2_2 = cv.medianBlur(input_salt_pepper, 3).astype(np.uint8)
output_2_3 = cv.medianBlur(input_salt_pepper, 5).astype(np.uint8)
output_2_4 = cv.medianBlur(input_salt_pepper, 7).astype(np.uint8)

#########################################################
#################### Weighted Median ####################
#########################################################

def weighted_median_filter(img,kernel_size,cen_weight):
    padding_size = kernel_size // 2
    img_padded = np.pad(img, (padding_size, padding_size), 'edge')
    center_weight = cen_weight
    output = np.zeros(img.shape)
    for i in range(padding_size,img_padded.shape[0]-padding_size):
        for j in range(padding_size,img_padded.shape[0]-padding_size):
            values = img_padded[i-padding_size:i+padding_size+1, j-padding_size:j+padding_size+1].flatten()
            values = np.append(values, [img_padded[i,j]] * center_weight)
            median = np.median(values)
            output[i-padding_size,j-padding_size] = median
    output = (output + 0.5) // 1
    return output

output_2_5 = weighted_median_filter(input_salt_pepper,3,3).astype(np.uint8)
output_2_6 = weighted_median_filter(input_salt_pepper,5,5).astype(np.uint8)
output_2_7 = weighted_median_filter(input_salt_pepper,7,7).astype(np.uint8)

#########################################################
#################### Printing ###########################
#########################################################

psnr_output_1_1 = str(cv.PSNR(output_1_1, ground_truth))
psnr_output_1_2 = str(cv.PSNR(output_1_2, ground_truth))
psnr_output_1_3 = str(cv.PSNR(output_1_3, ground_truth))
psnr_output_2_1 = str(cv.PSNR(output_2_1, ground_truth))
psnr_output_2_2 = str(cv.PSNR(output_2_2, ground_truth))
psnr_output_2_3 = str(cv.PSNR(output_2_3, ground_truth))
psnr_output_2_4 = str(cv.PSNR(output_2_4, ground_truth))
psnr_output_2_5 = str(cv.PSNR(output_2_5, ground_truth))
psnr_output_2_6 = str(cv.PSNR(output_2_6, ground_truth))
psnr_output_2_7 = str(cv.PSNR(output_2_7, ground_truth))

#########################################################
#################### Results ############################
#########################################################

print("PSNR of Gaussian Noisy Image: " + str(cv.PSNR(input_gaussian, ground_truth)))
print("PSNR of Gaussian Noisy Image with Adaptive Mean Filter: " + psnr_output_1_1)
print("PSNR of Gaussian Noisy Image with OpenCV 5x5 Box Filter: " + psnr_output_1_2)
print("PSNR of Gaussian Noisy Image with OpenCV 5x5 Gaussian Filter: " + psnr_output_1_3)
print("PSNR of Salt and Pepper Noisy Image: " + str(cv.PSNR(input_salt_pepper, ground_truth)))
print("PSNR of Salt and Pepper Noisy Image with Adaptive Median Filter: " + psnr_output_2_1)
print("PSNR of Salt and Pepper Noisy Image with OpenCV 3x3 Median Filter: " + psnr_output_2_2)
print("PSNR of Salt and Pepper Noisy Image with OpenCV 5x5 Median Filter: " + psnr_output_2_3)
print("PSNR of Salt and Pepper Noisy Image with OpenCV 7x7 Median Filter: " + psnr_output_2_4)
print("PSNR of Salt and Pepper Noisy Image with 3x3 Weighted Median Filter: " + psnr_output_2_5)
print("PSNR of Salt and Pepper Noisy Image with 5x5 Weighted Median Filter: " + psnr_output_2_6)
print("PSNR of Salt and Pepper Noisy Image with 7x7 Weighted Median Filter: " + psnr_output_2_7)


cv.imshow('Gaussian Noisy Image : ' + str(cv.PSNR(input_gaussian, ground_truth)), input_gaussian)
cv.imshow('Adaptive Mean Filter : ' + psnr_output_1_1,  output_1_1.astype(np.uint8))
cv.imshow('OpenCV 5x5 Box Filter : ' + psnr_output_1_2,  output_1_2.astype(np.uint8))
cv.imshow('OpenCV 5x5 Gaussian Filter : ' + psnr_output_1_3,  output_1_3.astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow('Salt and Pepper Noisy Image : ' + str(cv.PSNR(input_salt_pepper, ground_truth)), input_salt_pepper)  
cv.imshow('Adaptive Median Filter : ' + psnr_output_2_1,  output_2_1.astype(np.uint8))
cv.imshow('OpenCV 3x3 Median Filter : ' + psnr_output_2_2,  output_2_2.astype(np.uint8))
cv.imshow('OpenCV 5x5 Median Filter : ' + psnr_output_2_3,  output_2_3.astype(np.uint8))
cv.imshow('OpenCV 7x7 Median Filter : ' + psnr_output_2_4,  output_2_4.astype(np.uint8))
cv.imshow('Weighted 3x3 Median Filter : ' + psnr_output_2_5,  output_2_5.astype(np.uint8))
cv.imshow('Weighted 5x5 Median Filter : ' + psnr_output_2_6,  output_2_6.astype(np.uint8))
cv.imshow('Weighted 7x7 Median Filter : ' + psnr_output_2_7,  output_2_7.astype(np.uint8))


cv.waitKey(0)
cv.destroyAllWindows()