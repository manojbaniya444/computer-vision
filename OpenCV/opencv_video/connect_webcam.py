import cv2

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray_frame)
    
    if cv2.waitKey(100) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()