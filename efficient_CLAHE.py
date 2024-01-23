import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random


def is_corner(x,y,img,centers,tile_grid_size):
    # left top
    if x >= 0 and x < tile_grid_size[0]//2 and y >= 0 and y < tile_grid_size[0]//2:
        return True
    # left bottom
    elif x >= centers[centers.shape[0]-1][0] and x < img.shape[0] and y >= 0 and y < tile_grid_size[0]//2:
        return True
    # right top
    elif x >= 0 and x < tile_grid_size[0]//2 and y >= centers[centers.shape[0]-1][1] and y < img.shape[1]:
        return True
    # right bottom
    elif x >= centers[centers.shape[0]-1][0] and x < img.shape[0] and y >= centers[centers.shape[0]-1][1] and y < img.shape[1]:
        return True
    else:
        return False

def is_border(x,y,img,centers,tile_grid_size):
    # left border
    if x >= 0 and x < img.shape[0] and y >= 0 and y < tile_grid_size[0]//2:
        return True
    # right border
    elif x >= 0 and x < img.shape[0] and y >= centers[centers.shape[0]-1][1] and y < img.shape[1]:
        return True
    # top border
    elif x >= 0 and x < tile_grid_size[0]//2 and y >= 0 and y < img.shape[1]:
        return True
    # bottom border
    elif x >= centers[centers.shape[0]-1][0] and x < img.shape[0] and y >= 0 and y < img.shape[1]:
        return True
    else:
        return False

def find_tile(i,j,centers,tile_grid_size):
    for k in range(centers.shape[0]):
        if i >= centers[k][0] - tile_grid_size[0] // 2 and i < centers[k][0] + tile_grid_size[0] // 2 and j >= centers[k][1] - tile_grid_size[1] // 2 and j < centers[k][1] + tile_grid_size[1] // 2:
            return k
            
def get_nearest_tiles(img,tile_grid_size,i,j,centers):

    y_tiles = img.shape[1] // tile_grid_size[1]

    tile_num = find_tile(i,j,centers,tile_grid_size)
    #left top
    if i < centers[tile_num][0] and j < centers[tile_num][1]:

        # if the pixel is in the left border of the image
        if j < tile_grid_size[0] // 2:
            return np.array([tile_num,tile_num-y_tiles-1])
        # if the pixel is in the top border of the image
        elif i < tile_grid_size[1] // 2:
            return np.array([tile_num,tile_num-1])
        else:
            return np.array([tile_num,tile_num-1, tile_num-y_tiles, tile_num-y_tiles-1])
    
    # right top
    elif i < centers[tile_num][0] and j >= centers[tile_num][1]:
        # if the pixel is in the right border of the image
        if j >= img.shape[1] - tile_grid_size[1] // 2:
            return np.array([tile_num,tile_num-y_tiles+1])
        # if the pixel is in the top border of the image
        elif i < tile_grid_size[1] // 2:
            return np.array([tile_num,tile_num+1])
        else:
            return np.array([tile_num,tile_num+1, tile_num-y_tiles, tile_num-y_tiles+1])
    
    # left bottom
    elif i >= centers[tile_num][0] and j < centers[tile_num][1]:

        # if the pixel is in the left border of the image
        if j < tile_grid_size[0] // 2:
            return np.array([tile_num,tile_num+y_tiles])
        # if the pixel is in the bottom border of the image
        elif i >= img.shape[0] - tile_grid_size[0] // 2:
            return np.array([tile_num,tile_num-1])
        else:
            return np.array([tile_num,tile_num-1, tile_num+y_tiles, tile_num+y_tiles-1])
    
    # right bottom
    elif i >= centers[tile_num][0] and j >= centers[tile_num][1]:
        # if the pixel is in the right border of the image
        if j >= img.shape[1] - tile_grid_size[1] // 2:
            return np.array([tile_num,tile_num+y_tiles])
        # if the pixel is in the bottom border of the image
        elif i >= img.shape[0] - tile_grid_size[0] // 2:
            return np.array([tile_num,tile_num+1])
        else:
            return np.array([tile_num,tile_num+1, tile_num+y_tiles, tile_num+y_tiles+1])
    

def get_weights(i,j,centers,nearest_tiles,tile_grid_size):
    real_center = centers - 0.5
    # calculate the euclidean distance between the pixel and the center of each tile
    distances = np.array([np.sqrt((i - real_center[nearest_tiles[k]][0]) ** 2 + (j - real_center[nearest_tiles[k]][1]) ** 2) for k in range(nearest_tiles.shape[0])])
    # calculate the weights
    # if any the distance is 0, set the weight to 1
    weights = np.array([1 / distances[k] for k in range(distances.shape[0])])
    return weights / weights.sum()

def get_tile_center(img_x,img_y,num, tile_grid_size):
        row_num = img_x // tile_grid_size[0]
        col_num = img_y // tile_grid_size[1]
        row = num // col_num
        col = num % col_num
        return (row * tile_grid_size[0] + tile_grid_size[0] // 2, col * tile_grid_size[1] + tile_grid_size[1] // 2)

def efficient_CLAHE(img, clip_limit, tile_grid_size):
    orig_x = img.shape[0]
    orig_y = img.shape[1]
    output = np.zeros(img.shape)
    clip_limit = clip_limit * tile_grid_size[0] * tile_grid_size[1] / 256
    # if image cannot be divided into tiles, pad it
    if img.shape[0] % tile_grid_size[0] != 0:
        img = np.pad(img, ((0, tile_grid_size[0] - img.shape[0] % tile_grid_size[0]), (0, 0)), 'constant')
    if img.shape[1] % tile_grid_size[1] != 0:
        img = np.pad(img, ((0, 0), (0, tile_grid_size[1] - img.shape[1] % tile_grid_size[1])), 'constant')
    # divide image into tiles
    tiles = np.array([img[i:i + tile_grid_size[0], j:j + tile_grid_size[1]] for i in range(0, img.shape[0], tile_grid_size[0]) for j in range(0, img.shape[1], tile_grid_size[1])])
    
    # get the center of each tile
    centers = np.array([get_tile_center(img.shape[0], img.shape[1], i, tile_grid_size) for i in range(tiles.shape[0])])

    # calculate histogram of each tile
    hist = np.array([np.histogram(tiles[i], bins=256, range=(0, 256))[0] for i in range(tiles.shape[0])])
    cum_hist = np.zeros(hist.shape)
    for i in range(hist.shape[0]):
        limits = np.where(hist[i] > clip_limit)[0]
        # if there are any values greater than clip limit
        if limits.shape[0] > 0:
            limit_count = np.array([hist[i][limits[k]] - clip_limit for k in range(limits.shape[0])])
            over_count = np.sum(limit_count)
            non_limits = np.where(hist[i] <= clip_limit)[0]
            hist[i][limits] = clip_limit
            for l in range(int(over_count)):
                hist[i][random.choice (non_limits)] += 1

        g_min = 0
        for l in range(256):
            if hist[i][l] > 0:
                g_min = l
                break

        H_c = np.cumsum(hist[i])

        H_min = H_c[g_min]

        T = (( H_c - H_min ) / ((tile_grid_size[0]*tile_grid_size[1]) - H_min))*255

        Q = np.ceil(T - 0.5)

        cum_hist[i] = Q 


    for i in range(orig_x):
        for j in range(orig_y):
            # find the tile that contains the pixel
            if is_corner(i,j,img,centers,tile_grid_size):
                # if the pixel is a corner, use only 1 tile to calculate the transformation function
                # find the tile that contains the pixel
                tile_num = find_tile(i,j,centers,tile_grid_size)
                # calculate the transformation function
                output[i][j] = cum_hist[tile_num][img[i,j]]
            elif is_border(i,j,img,centers,tile_grid_size):
                # get two nearest tiles
                nearest_tiles = get_nearest_tiles(img,tile_grid_size,i,j,centers)[:2]
                # calculate weights
                weights = get_weights(i,j,centers,nearest_tiles,tile_grid_size)
                output[i,j] = cum_hist[nearest_tiles[0]][img[i,j]] * weights[0] + cum_hist[nearest_tiles[1]][img[i,j]] * weights[1]
            else:
                # get four nearest tiles
                nearest_tiles = get_nearest_tiles(img,tile_grid_size,i,j,centers)[:4]
                # calculate weights
                weights = get_weights(i,j,centers,nearest_tiles,tile_grid_size)
                # calculate the transformation function
                output[i,j] = cum_hist[nearest_tiles[0]][img[i,j]] * weights[0] + cum_hist[nearest_tiles[1]][img[i,j]] * weights[1] + cum_hist[nearest_tiles[2]][img[i,j]] * weights[2] + cum_hist[nearest_tiles[3]][img[i,j]] * weights[3]        
    return output


img = cv.imread('test6.jpg', 0)

out = efficient_CLAHE(img, 16, (32, 32))
out_cv = cv.createCLAHE(clipLimit=16, tileGridSize=(32, 32)).apply(img)

cv.imshow('Image 6', out.astype(np.uint8))
cv.imshow('Image 6 cv', out_cv.astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()


img2 = cv.imread('test7.jpg', 0)
out2 = efficient_CLAHE(img2, 16, (32, 32))
out2_cv = cv.createCLAHE(clipLimit=16, tileGridSize=(32, 32)).apply(img2)



cv.imshow('Image 7', out2.astype(np.uint8))
cv.imshow('Image 7 cv', out2_cv.astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()