import cv2

img = cv2.imread("./images/car2.jpg")

while True:
    cv2.imshow("image", img)
    #if we waited at least 1 ms AND we have pressed Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()