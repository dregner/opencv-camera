
import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture(0)
pts = 64

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([29, 86, 6])
    upper_green = np.array([64, 255, 255])
    lower_purple = np.array([130, 100,0])
    upper_purple =  np.array([160, 255, 255]) #mid = 150

   
     # Blur the image to have more pixels with same color
    blur = cv2.GaussianBlur(frame,(11, 11), 0)
    # blur = cv2.bilateralFilter(frame,9,75,75)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
  
     # Threshold the HSV image to get only blue colors
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    # mask_p = cv2.inRange(hsv, lower_purple, upper_purple)

    # construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask_t = mask_g #| mask_p
    kernel = np.ones((10,10),np.uint8)
    mask_t = cv2.erode(mask_t, kernel, iterations=1)
    mask_t = cv2.dilate(mask_t, kernel, iterations=1)

    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask_t)
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(mask_t, 127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)

    # parameters: frame, cnts_array, -1 = all, color,thickness
    # cv2.drawContours(frame, contours, -1, (0,255,0),3)

    if len(contours) > 0:
        cnt0 = contours[0]
        M = cv2.moments(cnt0)
        # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # cv2.circle(frame, center, 5, (255, 255, 255), -1)
        #one type of rectangles
        x,y,w,h = cv2.boundingRect(cnt0)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #Second type of rectangles
        # rect = cv2.minAreaRect(cnt1)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(frame,[box],0,(0,0,255),2)
        if len(contours) > 1:
            cnt1 = contours[1]
            M = cv2.moments(cnt1)
            # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # cv2.circle(frame, center, 5, (255, 255, 255), -1)
            (x,y),radius = cv2.minEnclosingCircle(cnt1)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(frame,center,radius,(0,0,255),2)



            # epsilon = 0.1*cv2.arcLength(cnt0,True)
            # approx = cv2.approxPolyDP(cnt0,epsilon,True)

    # cv2.imshow('frame',gaussian_blur)
    cv2.imshow('frame', frame)
    # cv2.imshow('blur', blur)
    cv2.imshow('mask', mask_t)
    # cv2.imshow('res',res)


    cv2.waitKey(2)

cv2.destroyAllWindows()
