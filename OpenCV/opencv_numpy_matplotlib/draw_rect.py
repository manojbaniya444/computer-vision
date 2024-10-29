import numpy as np
import cv2

# variables
drawing = False
ix,iy = -1,-1

# function
def draw(event,x,y,flags,param):
    global drawing,ix,iy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)

img = np.zeros((512,512,3))
cv2.namedWindow(winname="my_window")
cv2.setMouseCallback("my_window",draw)

while True:
    cv2.imshow("my_window", img)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break

        
cv2.destroyAllWindows()