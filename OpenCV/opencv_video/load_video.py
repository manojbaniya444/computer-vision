import cv2
import time

cap = cv2.VideoCapture('myVideo.mp4')
fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if cap.isOpened() == False:
    print("Error file not found:")
    
while cap.isOpened():
    ret,frame = cap.read()
    
    if ret:
        time.sleep(1/fps)
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break # close after the video ends

cap.release
cv2.destroyAllWindows()