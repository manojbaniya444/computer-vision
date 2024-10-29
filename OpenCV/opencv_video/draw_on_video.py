import cv2

cap = cv2.VideoCapture(0)

# variables
pt1 = (0,0)
pt2 = (0,0)
top_left_clicked = False
bottom_right_clicked = False

# draw 
def draw_rect(event, x, y, flags, param):
    global pt1, pt2, top_left_clicked, bottom_right_clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if top_left_clicked and bottom_right_clicked:
            pt1 = (0,0)
            pt2 = (0,0)
            top_left_clicked = False
            bottom_right_clicked = False
            
        if top_left_clicked == False:
            pt1 = (x,y)
            top_left_clicked = True
        elif bottom_right_clicked == False:
            pt2 = (x,y)
            bottom_right_clicked = True
            


cv2.namedWindow("frame")
cv2.setMouseCallback("frame",draw_rect)

while True:
    ret,frame = cap.read()
    
    if top_left_clicked:
        cv2.circle(frame,pt1,2,-1)
        
    if top_left_clicked and bottom_right_clicked:
        cv2.rectangle(frame,pt1,pt2,(0,255,0),2)
    
    
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()