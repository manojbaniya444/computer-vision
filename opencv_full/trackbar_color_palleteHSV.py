import cv2
import numpy as np

def nothing(x):
    pass

# create a window
cv2.namedWindow('image')

# capture video
cap = cv2.VideoCapture(0)

# image
image = cv2.imread("./images/green_car.jpeg")
image = cv2.resize(image, (300, 300))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# create trackbars
cv2.createTrackbar('H_low', 'image', 0, 179, nothing )
cv2.createTrackbar('S_low', 'image', 0, 255, nothing )
cv2.createTrackbar('V_low', 'image', 0, 255, nothing )

cv2.createTrackbar('H_high', 'image', 0, 179, nothing)
cv2.createTrackbar('S_high', 'image', 0, 255, nothing)
cv2.createTrackbar('V_high', 'image', 0, 255, nothing)

while(True):
    
    h_low = cv2.getTrackbarPos('H_low', 'image')
    s_low = cv2.getTrackbarPos('S_low', 'image')
    v_low = cv2.getTrackbarPos('V_low', 'image')
    
    h_high = cv2.getTrackbarPos('H_high', 'image') 
    s_high = cv2.getTrackbarPos('S_high', 'image')
    v_high = cv2.getTrackbarPos('V_high', 'image')

    lower_range = np.array([h_low, s_low, v_low])
    upper_range = np.array([h_high, s_high, v_high])

    mask = cv2.inRange(hsv, lower_range, upper_range)
    # frame = cv2.bitwise_and(image, image, mask = mask)

    cv2.imshow('image', mask)
    
    cv2.imshow("original", image)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    
cv2.destroyAllWindows()
