import cv2
from ultralytics import YOLO


# Load the YOLOv8 model
model = YOLO('best.pt')

names = model.model.names


# Open the video file
# video_path = "path/to/your/video/file.mp4"
cap = cv2.VideoCapture('rtsp://192.168.1.66:8080/h264_ulaw.sdp')


# Set the frame skip factor
frame_skip = 10
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # cv2.imshow("original frame", frame)

    frame_count += 1

    # Process every 5th frame
    if success:

        if frame_count % frame_skip == 0:

            frame = cv2.resize(frame, (400,400))
            original_frame = frame.copy()
            # cv2.rectangle(frame, (0, 300), (400, 400), (255, 255, 255), 2)

            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            for result in results:
                    boxes = result.boxes.xyxy.cpu()
                    clss = result.boxes.cls.cpu().tolist()
                    confs = result.boxes.conf.float().cpu().tolist()
                    annotated_frame = frame

                    annotated_frame = results[0].plot() if any(conf > 0.7 for conf in confs) else annotated_frame

                    cv2.imshow("YOLOv8 Inference", annotated_frame)

                    for box, cls, conf in zip(boxes, clss, confs):
                        print(f"Class Name: {names[int(cls)]}, Confidence Score: {conf}")
                        class_name = names[int(cls)]


                        # show the frame only when license plate is detected and when the confidence score is greater than 70 percent
                        if class_name == 'license_plate' and conf > 0.7:
                            print(f"Class Name: {class_name}, Confidence Score: {conf}, Bounding Box: {box}")
                            coordinates = box  # license plate coordinates

                            cropped_plate = original_frame[int(coordinates[1]):int(coordinates[3]), int(coordinates[0]):int(coordinates[2])]
                            cropped_plate = cv2.resize(cropped_plate, (300, 100))
                            cv2.imshow("Cropped License Plate", cropped_plate)
                            cv2.imwrite("cropped_plate.jpg", cropped_plate)
    

                            # if the license plate is inside the green rectangle then only perform the croping of the license plate
                            # if coordinates[0] > 0 and coordinates[1] > 300 and coordinates[2] < 400 and coordinates[3] < 400:
                            #     print("License plate cropped")
                            #     cropped_plate = original_frame[int(coordinates[1]):int(coordinates[3]), int(coordinates[0]):int(coordinates[2])]
                            #     cropped_plate = cv2.resize(cropped_plate, (300, 100))
                            #     cv2.imshow("Cropped License Plate", cropped_plate)

            # Break the loop if 'q' is pressed

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()