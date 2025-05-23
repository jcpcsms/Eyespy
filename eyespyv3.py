#############################===== +EYESPY V3.0, PAZ, J.C. FULL SAIL UNIVERSITY CSMS =====####################################################
# deps
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
import winsound 
import threading
from ultralytics import YOLO
####################################################################################################################################
# load Yolo models,  gun_model was provided by Felix Sam, https://github.com/Tech-Watt/Yolo11-Gun-Detection-Model/blob/main/model.pt
model = YOLO('data/yolo11n.pt')
gun_model= YOLO('data/model.pt') 
####################################################################################################################################
# Function 1. draw_bounding_boxes
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
####################################################################################################################################
# Function 2. id_faces & eyes
def id_faces(frame, face_cascade, eye_cascade):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
                save_detection_log(faces, (x, y) , "Face Detection")

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                if ew > 20 and eh > 20:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
                    cv2.putText(roi_color, "Eye", (ex, ey - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
                    save_detection_log(eyes, (ex, ey, ex + ew, ey + eh), "Eye Detection")

        return faces

    except Exception as e:
        messagebox.showerror("Detection I.D. Error", f"Face scan error: {str(e)}")
        return []
####################################################################################################################################
# Function 3. save_detection_log
def save_detection_log(label, coordinates, detection_type):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  
    log_filename = os.path.join(log_dir, "detection_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {detection_type}: {label} at {coordinates}\n"
    try:
        with open(log_filename, "a") as log_file:
            log_file.write(log_message)
        print(f"Logged: {log_message}")  
    except Exception as e:
        print(f"Error writing to log file: {str(e)}")       
####################################################################################################################################
# Function 4. cam_config
def cam_config():
    try:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise ValueError("Camera Error")
        return video_capture
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None
####################################################################################################################################
# Function 5. clean exit
def exit(video_capture):
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
    cv2.destroyAllWindows()
    root.destroy()
####################################################################################################################################
# Function 6. take_screenshot(including object detection)
def take_screenshot():
    ret, frame = video_capture.read()
    if not ret:
        messagebox.showerror("Screenshot failure")
        return
    try:
        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]}: {conf:.2f}"

                if conf > 0.61:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)
                   
        gun_results = gun_model(frame)
        for result in gun_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls_id = int(box.cls[0])
                label = f"{gun_model.names[cls_id]}: {conf:.2f}"

                if conf > 0.60:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
                    
        screenshot_dir = "screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Eyespy_capture_{timestamp}.png"
        filepath = os.path.join(screenshot_dir, filename)
        cv2.imwrite(filepath, frame)
        messagebox.showinfo("Eyespy", f"Screenshot saved as {filename}")
    except Exception as e:
        messagebox.showerror("Screenshot Error", f"Error capturing screenshot: {str(e)}")
####################################################################################################################################
# Function 7. nonblocking alert + play alarm sound
def show_nonblocking_alert(title, message):
    def show_alert():
        alert_window = tk.Toplevel(root)
        alert_window.title(title)
        alert_window.geometry("300x150")
        alert_window.configure(bg="red")  
        alert_window.attributes("-topmost", True)
        msg_label = tk.Label(alert_window, text=message, font=("Arial", 16), bg="red", fg="white", wraplength=280)
        msg_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        dismiss_button = tk.Button(alert_window, text="Dismiss", command=alert_window.destroy, font=("Arial", 12))
        dismiss_button.pack(pady=10)
        take_screenshot()
        alert_window.after(5000, alert_window.destroy)
        try:
            winsound.PlaySound('Sounds\\siren.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Sound error: {str(e)}")
    
    alert_thread = threading.Thread(target=show_alert)
    alert_thread.daemon = True 
    alert_thread.start()
      
alert_triggered = False
alert_label = None
##########################################################################################################################################
# Function 8. process_new_model_detections & triggrting alert Gun detection from public dataset
def process_new_model_detections(frame):
    global alert_triggered  
    
    try:
        results = gun_model(frame)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                try:
                    coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                    x1, y1, x2, y2 = map(int, coords)
                    print(f"Gun detection coordinates: ({x1}, {y1}, {x2}, {y2})")
                    
                    conf = float(box.conf[0]) if isinstance(box.conf, (list, tuple, np.ndarray)) else float(box.conf)
                    cls_id = int(box.cls[0]) if isinstance(box.cls, (list, tuple, np.ndarray)) else int(box.cls)
                    label = f"{gun_model.names[cls_id]}: {conf:.2f}"
                    
                    if conf > 0.4:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
                        save_detection_log(label, (x1, y1, x2-x1, y2-y1), "Weapon Model Detection")
                        #Alert trigger
                        if gun_model.names[cls_id] == 'Gun' and conf > 0.65 and not alert_triggered:
                            show_nonblocking_alert("EyeSpy ALERT!", "FIREARM DETECTED!")
                            alert_triggered = True
                            def reset_alert_flag():
                                global alert_triggered
                                alert_triggered = False
                            timer = threading.Timer(10.0, reset_alert_flag)
                            timer.daemon = True
                            timer.start()
    
                except Exception as e:
                    print(f"Error processing gun detection box: {str(e)}")
                    continue
        
        return frame
    except Exception as e:
        print(f"Error in gun model detection: {str(e)}")
        return frame
##########################################################################################################################################
# Function 9. upadte_frame
def update_frame():
    try:
        ret, frame = video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Check camera.")
            return

        id_faces(frame, faceCascade, eyeCascade)

        results = model(frame) 
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                label = f"{class_name}: {conf:.2f}"

                if conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 2)
                    save_detection_log(label, (x1, y1, x2-x1, y2-y1), "OBJECT DETECTIONS")    
                        
        frame = process_new_model_detections(frame)
        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_img)
        imgtk = ImageTk.PhotoImage(image=img)
        display_label.imgtk = imgtk
        display_label.configure(image=imgtk)
        display_label.after(500, update_frame)

    except Exception as e:
        messagebox.showerror("Frame Update Error", f"Error during frame update: {str(e)}")
################################################################################################################################
# Initialize the GUI application
root = tk.Tk()
root.title("Eyespy+")
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
