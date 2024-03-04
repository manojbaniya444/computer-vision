import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

from yolo_detect import YOLOModel
from yolo_predict import YOLOPredict 
from classify_class import ClassificationModel
from actions import add_image, add_license_image, run_image_detection, start_detection,add_video,stop_video,add_live_video

from db import create_database, fetch_latest_records

left_frame_color = "#1e1e1e"
right_frame_color = "#d0d0d0"
down_frame_color = "#8f8f8f"
navbar_color = "black"


file_path = ""

# Define global variables
down_frame = None
right_canvas = None 

# Add lists to store down canvases and labels
down_canvases = []
down_labels = []

#####_______________FUNCTIONS_________________#####
#? Adding new frame
def add_frame(container, frames, frame_func, frame_name):
    frame = frame_func(container)
    frames[frame_name] = frame
    # Configure row and column weights to make the frame expandable
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)
    # Use grid to make the frame take full width and height
    frame.grid(row=0, column=0, sticky="nsew")

#? Showing the required frame
def show_frame(frames, frame_name):
    frame = frames[frame_name]
    frame.tkraise()
    
#? Detect from image frame
def ObjectDetectionPage(parent):
    global down_frame, down_canvases, down_labels, right_canvas
    
    root = tk.Frame(parent, bg='blue')
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Left Frame
    left_frame = tk.Frame(root, bg=left_frame_color)
    left_frame.grid(row=0, column=0, sticky="nsew")

    # Right Frame
    right_frame = tk.Frame(root, bg=right_frame_color)
    right_frame.grid(row=0, column=1, sticky="nsew")
    
    # Configure row and column weights for resizing
    root.grid_columnconfigure(0, weight=2)
    root.grid_columnconfigure(1, weight=7)
    
    ## * In the right frame
    # Down Frame
    down_frame = tk.Frame(right_frame, bg=down_frame_color, height=90)
    down_frame.pack(side="bottom", fill="x")

    # Add canvas to the right frame
    right_canvas = tk.Canvas(right_frame, bg="#d0d0d0", width=640, height=640)
    right_canvas.pack(padx=15, pady=10, side="left")


    ##? To show the detected items from the database

    # right_right_frame = tk.Frame(right_frame, bg="green",width=400)
    # right_right_frame.pack(side="right", fill="y")

    # Create Treeview widget
    tree = ttk.Treeview(right_frame, style="Custom.Treeview")
    tree["columns"] = ("Plate Number", "Capture Date", "Vehicle Type")
    tree.column("#0", width=0, stretch=tk.NO)  # To hide the first empty column
    tree.column("Plate Number", anchor=tk.W, width=100)
    tree.column("Capture Date", anchor=tk.W, width=100)
    tree.column("Vehicle Type", anchor=tk.W, width=100)

    tree.heading("Plate Number", text="Plate Number")
    tree.heading("Capture Date", text="Capture Date")
    tree.heading("Vehicle Type", text="Vehicle Type")

    # Configure Treeview style
    tree_style = ttk.Style()
    tree_style.configure("Custom.Treeview", background="white", foreground="white", font=("Arial", 12))
    tree_style.configure("Custom.Treeview.Heading", background="#eeeeee", foreground="black", font=("Arial", 14))

    # Apply alternate colors to rows
    tree.tag_configure("row", background="#005b96")  # Apply row style
    tree.tag_configure("spacer", background="white")  # Apply spacer row style

    # Pack Treeview widget
    tree.pack(expand=True, fill=tk.BOTH, pady=20, padx=5)
    
    
    #_________comtents of left frame____________________#

    # Add content to left frame
    tk.Label(left_frame, text="Choose option", fg="white", bg=left_frame_color, font=("Arial", 17)).pack(pady=10)

    # Add buttons to the left frame
    label1 = tk.Label(left_frame, text="Choose Image", font=("Arial", 10), fg="white", bg=left_frame_color)
    btn1 = tk.Button(left_frame, text="Import Image", command=lambda: add_image(right_canvas), width=15, height=1)
    
    label3 = tk.Label(left_frame, text="Detect and Classify", font=("Arial", 10), bg=left_frame_color, fg="white")
    btn3 = tk.Button(left_frame, text="Detect", command=lambda: run_image_detection(down_canvases, down_labels, down_frame, right_canvas,tree), width=15,bg="#005b96",fg="white")

    # Pack widgets in left frame
    label1.pack(pady=5)
    btn1.pack(pady=5)
    label3.pack(pady=5)
    btn3.pack(pady=5)
    
    return root


#? Detect from video frame
def VideoObjectDetectionPage(parent):
    root = tk.Frame(parent, bg="#005b96")
    
    ##?__________________LAYOUT SETUP___________________##
    
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=6)
    root.grid_rowconfigure(2, weight=4)
    
    root.grid_columnconfigure(0, weight=3)
    root.grid_columnconfigure(1, weight=7)
    
    ##? for choosing video and showing live video
    option_frame = tk.Frame(root, bg="gray",height=90)
    option_frame.grid(row=0,columnspan=2,sticky="nsew")
    
    ##? For displaying live video
    live_frame = tk.Frame(root, bg ="#ececec")
    live_frame.grid(row=1,column=0,sticky="nsew")
    
   
    live_canvas = tk.Canvas(live_frame, bg="#ececec", width=350, height=350)
    live_canvas.place(x=15,y=30) 
    
    ##? For showinfg cropped from where license plate was detected
    show_detected_frame = tk.Frame(root, bg="#ffffff")
    show_detected_frame.grid(row=2,column=0,sticky="nsew")
    
    ##? for showing a cropped license plate
    captured_frame = tk.Frame(root, bg="#e1e1e1")
    captured_frame.grid(row=1,column=1,sticky="nsew")
    
    license_label = tk.Label(show_detected_frame, bg="#ffffff")
    license_label.place(x=70,y=10)
    
    license_canvas = tk.Canvas(show_detected_frame, bg="#ffffff", width=250, height=100)
    license_canvas.place(x=50,y=90)
    
    detected_canvas = tk.Canvas(captured_frame, bg="#e1e1e1", width=350, height=350)
    detected_canvas.place(x=200,y=20)
    
    ##? for database stored value
    table_frame = tk.Frame(root, bg="#ffffff")
    table_frame.grid(row=2,column=1,sticky="nsew")
    
    # Create Treeview widget
    tree = ttk.Treeview(table_frame, style="Custom.Treeview")
    tree["columns"] = ("License Number", "Date", "Vehicle Type")
    tree.column("#0", width=0, stretch=tk.NO)  # To hide the first empty column
    tree.column("License Number", anchor=tk.W, width=100)
    tree.column("Date", anchor=tk.W, width=100)
    tree.column("Vehicle Type", anchor=tk.W, width=100)

    tree.heading("License Number", text="License Number")
    tree.heading("Date", text="Date")
    tree.heading("Vehicle Type", text="Vehicle Type")
    
    # Apply alternate colors to rows
    tree.tag_configure("row", background="#005b96")  # Apply row style

    # Pack Treeview widget
    tree.place(x=0, y=0, relwidth=1, relheight=1)
    ##?_________________________ADDING ITEMS TO THE FRAMES_______________________##
    
    #? Option Frame
    video_choose_btn = tk.Button(option_frame, text="Choose Video", command=lambda: add_video(live_canvas), bg="gray", fg="white")
    video_choose_btn.pack(padx=10,side="left")
 
    entry_label = tk.Label(option_frame, text="RTSP address:", bg="gray")
    entry_label.pack(padx=10,side="left")
    # Add entry widget to the option frame
    entry_widget = tk.Entry(option_frame,width=60, bg="white")
    entry_widget.pack(padx=10,side='left')
    
    show_live_btn = tk.Button(option_frame, text="Show Live", command=lambda: add_live_video(entry_widget.get(),live_canvas), bg="gray", fg="white")
    show_live_btn.pack(padx=10,side="left")
    
    stop_frame_btn = tk.Button(option_frame, text="Stop", command=lambda :stop_video(live_canvas,license_canvas,detected_canvas,license_label), bg="#ff0000", fg="white")
    stop_frame_btn.pack(padx=5,side='left')
    
    start_detection_btn = tk.Button(option_frame, text="Start Detection", command=lambda: start_detection(live_canvas,captured_frame,show_detected_frame,license_canvas,detected_canvas,license_label,tree), bg="#005b96", fg="white")
    start_detection_btn.pack(padx=5,side='left')
    
    ##?_______________________TABLE FRAME_____________________
    # Create Treeview widget
    
    
    return root
  
#########_________________________________________#########
def create_ui(root):    
     #?NAVBAR______ Top Frame (full width, height=50)
    top_frame = tk.Frame(root, bg=navbar_color, height=30)
    top_frame.pack(fill="x")
    
    # Create navigation buttons
    btn_object_detection = tk.Button(top_frame, text="Object Detection", command=lambda: show_frame(frames, "ObjectDetectionPage"),bg='black',fg='white')
    btn_object_detection.pack(side="left", padx=5, pady=5)
    btn_video_detection = tk.Button(top_frame, text="Video Detection", command=lambda: show_frame(frames, "VideoObjectDetectionPage"),bg='black', fg='white')
    btn_video_detection.pack(side="left", padx=5, pady=5)
    
    ##______________________NAVAR END AREA______________________#####
    #######_______________________ROUTING______________________##########
    ##? container frame
    container = tk.Frame(root)
    container.pack(fill="both", expand=True)
    
    ##? Dictionary to hold different frames
    frames = {}
    
    ##? Initialize and add frames to the container
    add_frame(container, frames, ObjectDetectionPage, "ObjectDetectionPage")
    add_frame(container, frames, VideoObjectDetectionPage, "VideoObjectDetectionPage")
    
    
    #####_______________________ROUTING END_______________________#####

    ## ? Show the first frame
    show_frame(frames, "ObjectDetectionPage")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("License Plate Recognition System")
    
    print("Loading YOLO model...")
    model = YOLOModel()
    modelpredict = YOLOPredict()
    print("YOLO model loaded successfully")
    
    print("Loading classification model...")
    classification_model = ClassificationModel()
    print("Classification model loaded successfully")

    

    create_ui(root)
    
    create_database()
    
    root.mainloop()

