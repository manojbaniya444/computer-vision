import numpy as np
import cv2

def draw(event,x,y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,255,0),2)

img = np.zeros((512,512,3))
cv2.namedWindow(winname="draw")
cv2.setMouseCallback("draw",draw)

while True:
    cv2.imshow('draw', img)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()