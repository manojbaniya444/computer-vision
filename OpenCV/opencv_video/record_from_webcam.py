import cv2

capture = cv2.VideoCapture(0)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('myVideo.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 20,(width,height))


while True:
    ret, frame = capture.read()
    writer.write(frame)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
writer.release()
cv2.destroyALlWindows()