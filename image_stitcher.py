import numpy as np
import cv2 as cv

#Parameters
BRUTE_FORCE = False

# Options : 'SIFT' , 'ORB' , 'BRISK' 
FEATURE_EXTRACTOR = 'ORB'

# Read the image

img1 = cv.imread('uni_test_1.jpg')
img2 = cv.imread('uni_test_2.jpg')
img3 = cv.imread("uni_test_3.jpg")
img4 = cv.imread("uni_test_4.jpg")

#########################################################
########### Some Helper Functions #######################
#########################################################

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)


#########################################################
#################### Question 1 #########################
#########################################################


def stitch(img1, img2, method, bf):
    img1_color = img1
    img2_color = img2

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if method == 'BRISK':
        brisk = brisk = cv.BRISK_create()
        kp1, des1 = brisk.detectAndCompute(img1, None)
        kp2, des2 = brisk.detectAndCompute(img2, None)

        des1 = np.float32(des1)
        des2 = np.float32(des2)

    elif method == 'SIFT':
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

    elif method == 'ORB':
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        des1 = np.float32(des1)
        des2 = np.float32(des2)
    else:
        raise Exception('Invalid method')

    if bf == True:
        bf = cv.BFMatcher( crossCheck=True)
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda x: x.distance)
        
    else:
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(des1, des2, 2)


        good = []
        for m, n in knn_matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        matches = good
    
    # find homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
    # warp images
    result = cv.warpPerspective(img1_color, H, (img1_color.shape[1] + img2_color.shape[1], img1_color.shape[0]))
    result[0:img2_color.shape[0], 0:img2_color.shape[1]] = img2_color
    return result


res1 = stitch(img2, img1, FEATURE_EXTRACTOR, BRUTE_FORCE)
res2 = stitch(img3,img4, FEATURE_EXTRACTOR, BRUTE_FORCE)

# I found the four point transform function from the following link
# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def last_img(img):
    img_padded = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[0, 0, 0])
    img_orig = img.copy()

    img_padded = cv.cvtColor(img_padded, cv.COLOR_BGR2GRAY)

    thresh = cv.threshold(img_padded, 1, 255, cv.THRESH_BINARY)
    thresh = thresh[1]

    points = np.column_stack(np.where(thresh.transpose() > 0))
    hull = cv.convexHull(points)
    peri = cv.arcLength(hull, True)
    hullimg = img_orig.copy()
    hullimg = cv.polylines(hullimg, [hull], True, (0,255,0), 1)


    poly = cv.approxPolyDP(hull, 0.01 * peri, True)
    plist = poly.tolist()

    def order_points(pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    last_result = four_point_transform(img_orig, points)
    last_result = cv.cvtColor(last_result, cv.COLOR_BGR2RGB)

    return last_result

last_result1 = last_img(res1)
last_result2 = last_img(res2)

last_result1 = cv.cvtColor(last_result1, cv.COLOR_BGR2RGB)
last_result2 = cv.cvtColor(last_result2, cv.COLOR_BGR2RGB)

#########################################################
###################### Results  #########################
#########################################################

cv.imshow('Merge 1 and 2', ResizeWithAspectRatio(last_result1, height=720))
cv.imshow('Merge 3 and 4', ResizeWithAspectRatio(last_result2, height=720))

cv.waitKey(0)
cv.destroyAllWindows()
