# +EYESPY OBJECT DETECTION EXPIRIMENT, PAZ, J.C. FULL SAIL UNIVERSITY CSMS ##########
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
import winsound 
#####################################################################################
# Function 1. load_yolo_pt_model
def load_yolo_pt_model(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers
###################################################################################### 
# Function 2. draw_bounding_boxes
def draw_bounding_boxes(detections, classes, frame, width, height, confidence_threshold=20.):
    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = "{}: {:.2f}%".format(classes[class_id], confidence * 100)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)
            # Save detection log
            save_detection_log(label, (x, y, w, h), "Object Detection")
######################################################################################
# Function 3. id_faces
def id_faces(frame, face_cascade, eye_cascade):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            if w > 50 and h > 50:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)

            # Eye detection within the face region
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                if ew > 20 and eh > 20:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
                    cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)

        return faces

    except Exception as e:
        messagebox.showerror("Detection I.D. Error", f"Face scan error: {str(e)}")
        return []
######################################################################################
# Function 4. save_detection_log
def save_detection_log(label, coordinates, detection_type):
    log_filename = "detection_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {detection_type}: {label} at {coordinates}\n"
    with open(log_filename, "a") as log_file:
        log_file.write(log_message)
######################################################################################
# Function 5. cam_config
def cam_config():
    try:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise ValueError("Camera Error")
        return video_capture
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None
######################################################################################
# Function 6. exit
def exit(video_capture):
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
    cv2.destroyAllWindows()
    root.destroy()
######################################################################################
# Function 7. take_screenshot(including object detection)
def take_screenshot():
    ret, frame = video_capture.read()
    if not ret:
        messagebox.showerror("Screenshot failure")
        return
    # Face and eyes detections
    id_faces(frame, faceCascade, eyeCascade)
    # YOLO Object Detection
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detections = []
    for out in outs:
        for detection in out:
            detections.append(detection)

    draw_bounding_boxes(detections, classes, frame, width, height, confidence_threshold=0.61)
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Eyespy_capture_{timestamp}.png"
    filepath = os.path.join(screenshot_dir, filename)
    cv2.imwrite(filepath, frame)
    messagebox.showinfo("Eyespy", f"Screenshot saved as {filename}")

######################################################################################
# Function 8. update_frame updated with Alert Sound and Alert on detection Feature****
def play_alert_sound():
    sound_sample ='Sounds\\siren.wav'
    try:
        winsound.PlaySound(sound_sample, winsound.SND_FILENAME)
    except Exception as e:
        messagebox.showerror("EyeSpy", f"Error playing sound: {str(e)}")

def alert_on_detection(detections, classes, threshold=0.65, target_class='cell phone'):
    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > threshold and classes[class_id].lower() == target_class.lower():
            messagebox.showinfo("EyeSpy ALERT!", f"{target_class.title()} detected!")
            play_alert_sound()
            break 

def update_frame():
    try:
        ret, frame = video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Check camera.")
            return
        
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        detections = []
        for out in outs:
            for detection in out:
                detections.append(detection)

        draw_bounding_boxes(detections, classes, frame, width, height, confidence_threshold=0.61)

        alert_on_detection(detections, classes)

        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_img)
        imgtk = ImageTk.PhotoImage(image=img)

        display_label.imgtk = imgtk
        display_label.configure(image=imgtk)

        display_label.after(500, update_frame)

    except Exception as e:
        messagebox.showerror("Frame Update Error", f"Error during frame update: {str(e)}")
######################################################################################
# Initialize the GUI application
root = tk.Tk()
root.title("Face and YOLO Detection")
######################################################################################
# Create and configure the display label for the video
display_label = ttk.Label(root)
display_label.pack()
######################################################################################
# Add buttons for additional functionality
screenshot_button = ttk.Button(root, text="Take Screenshot", command=take_screenshot)
screenshot_button.pack(side=tk.LEFT)
######################################################################################
exit_button = ttk.Button(root, text="Exit", command=lambda: exit(video_capture))
exit_button.pack(side=tk.RIGHT)
######################################################################################
# Load the face cascade classifier
cascPath = os.path.join(os.path.dirname(cv2.__file__), "data/haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascPath)
if faceCascade.empty():
    messagebox.showerror("Error", "Check OpenCV version")
######################################################################################
# Load the eye cascade classifier
eyeCascadePath = os.path.join(os.path.dirname(cv2.__file__), "data/haarcascade_eye.xml")
eyeCascade = cv2.CascadeClassifier(eyeCascadePath)
if eyeCascade.empty():
    messagebox.showerror("Error", "Failed to load eye cascade.")
######################################################################################
# Load COCO dataset for YOLO
cfg_file = "data/yolov3.cfg"
weights_file = "data/yolov3.weights"
names_file = "data/coco.names"
net, classes, output_layers = load_yolo_pt_model(cfg_file, weights_file, names_file)
######################################################################################
# Setup video capture
video_capture = cam_config()
######################################################################################
# Start the frame update when camera is active
if video_capture is not None:
    update_frame()
######################################################################################
# Start the Tkinter event loop calling exit function as lambda
root.protocol("WM_DELETE_WINDOW", lambda: exit(video_capture))
root.mainloop()


# COCO Dataset Credit
#@misc{lin2015microsoft,
#      title={Microsoft COCO: Common Objects in Context},
#      author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
#      year={2015},
#      eprint={1405.0312},
#      archivePrefix={arXiv},
#      primaryClass={cs.CV}