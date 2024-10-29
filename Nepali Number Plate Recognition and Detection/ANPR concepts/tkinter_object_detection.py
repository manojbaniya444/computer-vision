import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading

class YOLOImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Detection App")

        # Create GUI elements
        self.choose_button = ttk.Button(root, text="Choose Image", command=self.choose_image)
        self.choose_button.pack(pady=10)

        self.progressbar = ttk.Progressbar(root, mode="indeterminate")
        self.image_label = ttk.Label(root)
        self.image_label.pack()

        # License plate image label
        self.license_plate_label = ttk.Label(root)
        self.license_plate_label.pack()

        # Save button
        self.save_button = ttk.Button(root, text="Save Image", command=self.save_image)
        self.save_button.pack()

        # Initialize variables
        self.image_with_boxes = None
        self.license_plate_image = None

        # Load the YOLOv8 model in the background during app initialization
        self.load_yolo_model()

    def load_yolo_model(self):
        self.model = YOLO('best.pt')
        self.hide_loading()

    def choose_image(self):
        # Reset the state before processing a new image
        self.reset_state()

        file_path = filedialog.askopenfilename(title="Choose an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.show_loading()
            threading.Thread(target=self.process_image, args=(file_path,), daemon=True).start()

    def reset_state(self):
        # Clear previous image and associated data
        self.image_with_boxes = None
        self.license_plate_image = None

        # Clear image labels
        self.image_label.configure(image=None)
        self.license_plate_label.configure(image=None)

    def show_loading(self):
        self.choose_button["state"] = "disabled"
        self.progressbar.start()
        self.progressbar.pack()

    def hide_loading(self):
        self.choose_button["state"] = "normal"
        self.progressbar.stop()
        self.progressbar.pack_forget()

    def process_image(self, file_path):
        # Read the selected image
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 inference on the image
        results = self.model(image)

        # Check if any detection has confidence greater than 70 percent
        if any(conf > 0.7 for result in results for conf in result.boxes.conf.float().cpu().tolist()):
            # Visualize the results on the image
            for result in results:
                self.image_with_boxes = result.plot()

            # Get the license plate coordinates with the highest confidence
            license_plate_coordinates = self.get_license_plate_coordinates(results)

            # Crop license plate if coordinates are available
            if license_plate_coordinates is not None:
                self.license_plate_image = self.crop_license_plate(image, license_plate_coordinates)
                self.display_license_plate(self.license_plate_image)
            else:
                self.display_no_detection_license_plate()

            # Resize the image with bounding boxes to be 640 pixels wide
            self.image_with_boxes = cv2.resize(self.image_with_boxes, (640, int(640 / image.shape[1] * image.shape[0])))

            # Convert the image to PhotoImage format for Tkinter
            img = Image.fromarray(self.image_with_boxes)
            img = ImageTk.PhotoImage(image=img)

            # Update the label with the new image
            self.image_label.img = img
            self.image_label.configure(image=img)
        else:
            # No detection with confidence > 70%, display a message or handle as needed
            self.display_no_detection()

        self.hide_loading()

    def get_license_plate_coordinates(self, results):
        max_confidence = 0
        best_coordinates = None

        for result in results:
            boxes = result.boxes.xyxy.cpu()
            clss = result.boxes.cls.cpu().tolist()
            confs = result.boxes.conf.float().cpu().tolist()

            for box, cls, conf in zip(boxes, clss, confs):
                class_name = self.model.model.names[int(cls)]

                if class_name == 'license_plate' and conf > 0.7:
                    if conf > max_confidence:
                        max_confidence = conf
                        best_coordinates = box

        return best_coordinates

    def crop_license_plate(self, image, coordinates):
        cropped_plate = image[int(coordinates[1]):int(coordinates[3]), int(coordinates[0]):int(coordinates[2])]
        cropped_plate = cv2.resize(cropped_plate, (300, 100))
        return cropped_plate

    def display_license_plate(self, license_plate_image):
        img = Image.fromarray(license_plate_image)
        img = ImageTk.PhotoImage(image=img)

        # Update the label with the license plate image
        self.license_plate_label.img = img
        self.license_plate_label.configure(image=img)

    def display_no_detection_license_plate(self):
        # Display a placeholder image or text when no license plate is detected
        text = "No License Plate Detected"
        label = ttk.Label(self.root, text=text)
        label.pack()
        
        # Display a placeholder image
        # You can replace this with your own placeholder image
        placeholder_img = Image.new("RGB", (300, 100), color=(255, 255, 255))
        placeholder_img = ImageTk.PhotoImage(image=placeholder_img)

        self.license_plate_label.img = placeholder_img
        self.license_plate_label.configure(image=placeholder_img)

    def display_no_detection(self):
        # Display a placeholder image or text when no detection is found
        text = "No Object Detected"
        label = ttk.Label(self.root, text=text)
        label.pack()

        # Display a placeholder image
        # You can replace this with your own placeholder image
        placeholder_img = Image.new("RGB", (640, 480), color=(255, 255, 255))
        placeholder_img = ImageTk.PhotoImage(image=placeholder_img)

        self.image_label.img = placeholder_img
        self.image_label.configure(image=placeholder_img)

    def save_image(self):
        if self.image_with_boxes is not None:
            # Convert the image to PIL format
            pil_image = Image.fromarray(self.image_with_boxes)

            # Save the image
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                pil_image.save(file_path)
                print(f"Image saved at {file_path}")
        else:
            print("No image to save. Please choose an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOImageApp(root)
    root.mainloop()
