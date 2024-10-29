import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tkinter import messagebox

from yolo_detect import YOLOModel
from yolo_predict import YOLOPredict
from utils import get_license_plate_coordinates, update_records
from segment import segment_and_classify

from db import insert_license_plate,fetch_latest_records

left_frame_color = "#1e1e1e"
right_frame_color = "#d0d0d0"
down_frame_color = "#8f8f8f"
navbar_color = "black"

# Image on the right canvas which is to be detected.
image_to_detect = None
image_to_detect_file = ""
frame_count = 0

cap = None
live_cap = None

START_VIDEO = False

model = YOLOPredict()
names = model.model.names

model2 = YOLOModel()

captured_ids = set()

processing_flag = False


def add_image(right_canvas, detected_image=None):
    global image_to_detect, image_to_detect_file
    # Show the image: detected_image if provided, else the original image
    if detected_image is not None:
        # Clear the existing content on the canvas
        right_canvas.delete("all")
        
        right_canvas.detected_image = detected_image
        right_canvas.create_image(0, 0, image=detected_image, anchor='nw')
        
    else:
        image_to_detect_file = filedialog.askopenfilename(initialdir='renamedimages', title='Select Image to Detect', filetypes=[('Image Files', '*.jpg *.png')])
        image = Image.open(image_to_detect_file)
        width, height = 640, 640
        image = image.resize((width, height), Image.LANCZOS)
        image_to_detect = ImageTk.PhotoImage(image)
        
        # Clear the existing content on the canvas
        right_canvas.delete("all") 
        
        right_canvas.image_to_detect = image_to_detect
        right_canvas.create_image(0, 0, image=image_to_detect, anchor='nw')

def add_license_image(down_canvases, down_labels, down_frame, detected_image=None,license_characters=""):
    # Create a new canvas and label for each call
    if detected_image is None:
        print("No image to add to the down frame.")
        # Create a label for the new canvas
        label_text = f"No license plate found."
        return
    
    if len(down_canvases) == 6:
        # Remove all canvases and labels if there are already 5
        for canvas, label in zip(down_canvases, down_labels):
            canvas.destroy()
            label.destroy()
        down_canvases.clear()
        down_labels.clear()

    # Calculate the index for placing the new canvas
    index = len(down_canvases)

    # Create a label for the new canvas

    label_text = license_characters
    canvas_label = tk.Label(down_frame, text=label_text, font=("Arial", 17), bg=down_frame_color)
    canvas_label.grid(row=0, column=index, padx=5, pady=3)

    # Create a new canvas and store it in the list
    canvas = tk.Canvas(down_frame, bg="white", width=150, height=60)
    canvas.grid(row=1, column=index, padx=5, pady=3)
    down_canvases.append(canvas)
    down_labels.append(canvas_label)

    # Configure column weights for equal horizontal distribution
    for i in range(len(down_canvases)):
        down_frame.grid_columnconfigure(i, weight=1)

    # Set the image on the new canvas
    # Resize the detected image
    canvas.image = detected_image
    canvas.create_image(0, 0, image=detected_image, anchor='nw')

def run_image_detection(down_canvases, down_labels, down_frame, right_canvas,tree):
    model = YOLOPredict()
    if image_to_detect is not None:
        print("Running detection on the image.")
        
        img_to_yolo = cv2.imread(image_to_detect_file)
        img_to_yolo_RGB = cv2.cvtColor(img_to_yolo, cv2.COLOR_BGR2RGB)
        
        results = model.predict(img_to_yolo_RGB)
        
        # Check if any detection has confidence greater than 70 percent
        if any(conf > 0.5 for result in results for conf in result.boxes.conf.float().cpu().tolist()):
            # Visualize the results on the image
            for result in results:
                annotated_image = result.plot()
                # ?Here the annotated image is in BGR format so converting it to RGB to supply to tkinter
                # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                # For tkinter supporting image
                annotated_image = cv2.resize(annotated_image, (640, 640))
                img = Image.fromarray(annotated_image)
                img = ImageTk.PhotoImage(image=img)
                # cv2.imshow("Result", annotated_image)
                add_image(right_canvas, img)
                
                # get license plate coordinates
                license_coordinates = get_license_plate_coordinates(results)
                
                if license_coordinates is not None:
                    x1,y1,x2,y2 = license_coordinates
                
                    # crop the license plate
                    cropped_license_plate = img_to_yolo[int(y1):int(y2), int(x1):int(x2)]
                    
                    # cv2.imwrite(f'license.jpg', cropped_license_plate)
                    
                    cropped_license_plate_RGB = cv2.cvtColor(cropped_license_plate, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(f'license_rgb.jpg', cropped_license_plate)
                    
                    #get character from License plate
                    license_characters = ""
                    characters_list,segmented_image = segment_and_classify(cropped_license_plate)
                    if len(characters_list) > 0:
                        license_characters = "".join(str(char) for char in characters_list)
                        ##? saving the information in the database
                        insert_license_plate(license_characters, cropped_license_plate)
                        update_records(tree)
                    
                    resized_cropped_license_plate = cv2.resize(cropped_license_plate_RGB, (150,60))
                
                    # pass the cropped license plate to add_license_image
                    license_plate = Image.fromarray(resized_cropped_license_plate)
                    license_plate = ImageTk.PhotoImage(image=license_plate)
                    print(license_characters)
                    add_license_image(down_canvases, down_labels, down_frame, license_plate,license_characters)
                    
                    
                else:
                    print("No license plate found.")
            print("Detection completed.")
                       
    else:
        print("Please select an image first to detect.")

#####_________________________________FOR THE VIDEO DETECTION PAGE ACTIONS___________________________#########
def stop_video(live_canvas,license_canvas,detected_canvas,license_label):
    global stop_flag
    
    # stop_flag = True
    if cap is not None:
        cap.release()
        stop_flag = True
        live_canvas.delete("all")
        license_canvas.delete("all")
        detected_canvas.delete("all")
        license_label.config(text="")
    if live_cap is not None:
        live_cap.release()
        stop_flag = True
        live_canvas.delete("all")
        print("Stream stopped")

def play_video(live_canvas,show_detected_frame,captured_frame,license_canvas, detected_canvas,license_label,tree):
    global stop_flag,START_VIDEO
    print("Inthe play video function")
    
    ret, frame = cap.read()
    # frame_count += 1
    print("Cap read")
    if ret and not stop_flag and START_VIDEO:
        
        if True:
            print("Start capturing")
            frame_cpy = frame.copy()
            to_model_resized = cv2.resize(frame, (640, 640))
            results = model2.track(to_model_resized)
        
            # Check if tracking IDs are available
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()
                annoted_frame = results[0].plot()            
                print(ids)

                # Draw boxes and IDs on the frame
                for box, id, cls , conf in zip(boxes, ids, classes, confs):
                    x1, y1, x2, y2 = box
                    class_name = names[int(cls)]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # capture the frame once when its center is at the right of the frame
                    if cy > 450 and id not in captured_ids and class_name == "license_plate" and conf > 0.6:
                        captured_ids.add(id)

                        ##? This cropped frame is the cropped license plate image                       
                        cropped_frame = to_model_resized[y1:y2, x1:x2]
                        print(f"License no. {id} captured.") 
                        # cv2.imshow(f"license{id}", cropped_frame)
                        
                        ##? Update the UI with detected license plate Image and Cropped Frame
                        display_licenseplate_frame(cropped_frame, annoted_frame,captured_frame,show_detected_frame,license_canvas, detected_canvas)
                        displate_detected_characters("No detection",license_label)
                        
                        ##? Display this license plate with detected number below in the show_detected_frame
                        license_characters = ""
                        characters_list,segmented_image = segment_and_classify(cropped_frame)
                        if len(characters_list) > 5:
                            license_characters = "".join(str(char) for char in characters_list)
                            #TODO:
                            # cv2.imwrite(f'licens{license_characters}.jpg', segmented_image)

                            ##? Update the UI with detected license plate Image
                            displate_detected_characters(license_characters,license_label)
                            insert_license_plate(license_characters, cropped_frame)
                            
                            
                            ##TODO: update the data in tree for the video frame
                            update_records(tree)
                        else:
                            displate_detected_characters("No detection",license_label)
                        
                ##? Show the original frame to the canvas
                frame_cpy = cv2.rectangle(to_model_resized, (20,400),(570,630),(37,245,99),2)
                frame = cv2.cvtColor(frame_cpy, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (350,350))
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(image=frame)

                # Clear previous frame if it exists
                live_canvas.delete("all")

                # Display the new frame on the canvas
                live_canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                live_canvas.image = frame                   
            else:
                print("Tracking IDs not available.")
            # Schedule the next frame
            if not stop_flag:
                live_canvas.after(60, lambda: play_video(live_canvas,show_detected_frame,captured_frame,license_canvas,detected_canvas,license_label,tree))
    else:
        print("Stream stopped.....")
        cap.release()
        stop_flag = True
        live_canvas.delete("all")
        print("Stream stopped")

def add_video(live_canvas):
    global cap, stop_flag
    stop_flag = False
    file_path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(file_path)
    
    
    ##_________________________LIVE STREAMING___________________________##
    
def add_live_video(rtsp_address, live_canvas):
    global stop_flag
    print("Trying to connect to stream...")

    if rtsp_address == "" or len(rtsp_address) < 30:
        messagebox.showwarning(title='No RTSP PROVIDED', message='Please provide the RTSP address')
        return
    else:
        messagebox.showinfo(title='RTSP PROVIDED', message='RTSP address provided')
        handle_video_capture(rtsp_address, live_canvas)

def handle_video_capture(rtsp_address, live_canvas):
    global stop_flag, live_cap,frame_count
    stop_flag = False

    try:
        print("Opening video from RTSP...")
        live_cap = cv2.VideoCapture(rtsp_address)
        
        if not live_cap.isOpened():
            raise RuntimeError("Failed to open RTSP stream")

        print("Trying to open the stream...")
        
        play_live_video(live_canvas, live_cap)

    except Exception as e:
        print("Error:", e)
        messagebox.showerror(title="Error", message="Failed to open the RTSP stream")
        
def play_live_video(live_canvas, cap):
    # Define the function for updating the canvas asynchronously
    global stop_flag
    def update_canvas():
        if not stop_flag:
            ret, frame = cap.read()  
            if ret:
                    # print("Stream started.....")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (350, 350))
                    frame = Image.fromarray(frame)
                    frame = ImageTk.PhotoImage(image=frame)

                    # Clear previous frame if it exists
                    live_canvas.delete("all")

                    # Display the new frame on the canvas
                    live_canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                    live_canvas.image = frame

                    # Schedule the next frame update
                    live_canvas.after(100, update_canvas)
            else:
                print("Stream stopped.....")
                cap.release()

    # Start the asynchronous update
    update_canvas()


## *show_detected_frame and captured_frame naming eta uta vko chaa
def start_detection(live_canvas,show_detected_frame,captured_frame,license_canvas, detected_canvas,license_label,tree):
    global START_VIDEO
    START_VIDEO = True
    print("Trying to detect")
    play_video(live_canvas,show_detected_frame,captured_frame,license_canvas,detected_canvas,license_label,tree)


def display_licenseplate_frame(license_plate, full_image, captured_frame, show_detected_frame,license_canvas, detected_canvas):
    # Resize the images
    license_plate = cv2.resize(license_plate, (250, 100))
    full_image = cv2.resize(full_image, (350, 350))

    # Convert the images to RGB format
    license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)
    full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

    # Convert the images to PIL format
    license_plate = Image.fromarray(license_plate)
    full_image = Image.fromarray(full_image)

    # Convert the images to ImageTk format
    license_plate = ImageTk.PhotoImage(image=license_plate)
    full_image = ImageTk.PhotoImage(image=full_image)

    ##? Create a canvas to show in the captured frame and show_detected_frame
    
    # destroy the previous canvas
    license_canvas.delete("all")
    detected_canvas.delete("all")
    
    # Display the new images on the canvases
    license_canvas.create_image(0, 0, image=license_plate, anchor='nw')
    detected_canvas.create_image(0, 0, image=full_image, anchor='nw')

    # Ensure the images are stored to prevent garbage collection
    license_canvas.image = license_plate
    detected_canvas.image = full_image


def displate_detected_characters(license_characters,license_label):
    license_label.config(text=license_characters, font=("Arial", 25))


