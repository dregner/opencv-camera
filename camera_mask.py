# import numpy as np
# import cv2

# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     print (gray)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#Function to increase brighness
def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

while(1):

    # Take each frame
    _, frame = cap.read()

    #iluminate image
    gamma = 1.5
    frame = adjust_gamma(frame, gamma=gamma)

    # Convert BGR to HSV
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50]) # mid = 120
    upper_blue = np.array([130,255,255])
    lower_green = np.array([29, 50, 50])
    upper_green = np.array([64, 255, 255])
    lower_red = np.array([0, 100, 100]) # mid = 20
    upper_red =  np.array([10, 255, 255])
    lower_purple = np.array([130, 100,0])
    upper_purple =  np.array([160, 255, 255]) #mid = 150

   
     # Blur the image to have more pixels with same color
    gaussian_blur = cv2.GaussianBlur(frame,(21, 21), 0)
    hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)

     # Threshold the HSV image to get only blue colors
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask_r = cv2.inRange(hsv, lower_red, upper_red)
    mask_p = cv2.inRange(hsv, lower_purple, upper_purple)

    mask_t = mask_g | mask_p
    kernel = np.ones((10,10), np.uint8)
    mask_t = cv2.erode(mask_t, kernel, iterations=1)
    mask_t = cv2.dilate(mask_t, kernel, iterations=1)

    # ret, threshol = cv2.adaptiveThreshold(mask_t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, threshol = cv2.threshold(gray,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # ret, threshol = cv2.threshold(gray,10, 255,cv2.THRESH_OTSU)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask_t)

    cv2.imshow('frame',frame)
    # cv2.imshow('threshold', threshol)
    cv2.imshow('mask', mask_t)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

    # masks determine value for color in HSV
# $ python
# >>> import cv2
# >>> import numpy as np  
# >>> green = np.uint8([[[0,255,0 ]]])
# >>> hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
# >>> print hsv_green
