import numpy as np
import cv2

drawing = False # true if mouse is pressed
mode = True # If true, draw rectangle toggle when 'm' is chosen

ix, iy = -1, -1

# mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing, mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 2,  (0,0,200), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv2.circle(img,(x, y), 2,(0,0,200), -1)
            
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    
    elif k == ord('n'):
        img = np.zeros((512, 512, 3), np.uint8)
    elif k == ord('q'):
        break
    
    
cv2.destroyAllWindows()
