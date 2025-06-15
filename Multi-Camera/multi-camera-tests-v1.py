#################### Multi-Camera Frame Test, SANS action recognition####################################
import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime
import winsound 
import threading
from ultralytics import YOLO
import torch
import requests
import glob
import os
import queue
import time
####################################################################################################################################
#load YOLO gun model 
gun_model= YOLO('data/model.pt')
#force YOLO to use GPU
if torch.cuda.is_available():
    gun_model.to('cuda')
    print(f"YOLO Gun Model moved to GPU")

#check if YOLO is using GPU
print(f"YOLO Gun Model Device: {gun_model.device}")
if torch.cuda.is_available():
    print(f"CUDA Available: {torch.cuda.device_count()} GPU(s)")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available - using CPU") 

####################################################################################################################################
# Camera selection - use USB, HTTP, RTSP, or video files for testing 
class Camera_Source_Selection:
    def __init__(self, parent, camera_manager):
        self.parent = parent
        self.camera_manager = camera_manager
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Camera Source Selection")
        self.dialog.geometry("600x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.camera_configs = {}
        self.create_interface()
        
    def create_interface(self):
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title_label = ttk.Label(main_frame, text="Configure Camera Sources", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))

        self.create_camera_config('priority_1', 'Priority Camera 1', main_frame)
        self.create_camera_config('priority_2', 'Priority Camera 2', main_frame)
        self.create_camera_config('object_1', 'Object Camera 1', main_frame)
        self.create_camera_config('object_2', 'Object Camera 2', main_frame)
        self.create_camera_config('object_3', 'Object Camera 3', main_frame)
    
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Start Selected Cameras", command=self.start_cameras).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT)
    
    def create_camera_config(self, camera_id, display_name, parent):
        frame = ttk.LabelFrame(parent, text=display_name)
        frame.pack(fill=tk.X, pady=2)
        
        self.camera_configs[camera_id] = {
            'enabled': tk.BooleanVar(value=False),
            'source_type': tk.StringVar(value='usb'),
            'source_path': tk.StringVar(value='0')
        }

        enable_cb = ttk.Checkbutton(frame, text="Enable", variable=self.camera_configs[camera_id]['enabled'])
        enable_cb.pack(side=tk.LEFT, padx=5)

        ttk.Label(frame, text="Type:").pack(side=tk.LEFT, padx=(10, 0))
        type_combo = ttk.Combobox(frame, textvariable=self.camera_configs[camera_id]['source_type'], 
                                 values=['usb', 'file', 'rtsp', 'http'], width=8, state='readonly')
        type_combo.pack(side=tk.LEFT, padx=5)
 
        ttk.Label(frame, text="Source:").pack(side=tk.LEFT, padx=(10, 0))
        source_entry = ttk.Entry(frame, textvariable=self.camera_configs[camera_id]['source_path'], width=30)
        source_entry.pack(side=tk.LEFT, padx=5)
   
        def browse_file():
            if self.camera_configs[camera_id]['source_type'].get() == 'file':
                filename = tk.filedialog.askopenfilename(
                    title=f"Select video file for {display_name}",
                    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
                )
                if filename:
                    self.camera_configs[camera_id]['source_path'].set(filename)
        
        ttk.Button(frame, text="Browse", command=browse_file, width=8).pack(side=tk.LEFT, padx=5)
    
    def start_cameras(self):
        self.camera_manager.stop_all_cameras()
        self.camera_manager.cameras.clear()
        
        for camera_id, config in self.camera_configs.items():
            if config['enabled'].get():
                source_type = config['source_type'].get()
                source_path = config['source_path'].get()
                
                if source_path.strip(): 
                    role = 'priority' if camera_id.startswith('priority') else 'object'
                    camera_config = {'loop': True} if source_type == 'file' else {}
                    
                    self.camera_manager.add_camera(camera_id, source_type, source_path, role, camera_config)
        
        success_count = self.camera_manager.start_all_cameras()
        
        if success_count > 0:
            messagebox.showinfo("Success", f"Started {success_count} cameras")
            self.dialog.destroy()
        else:
            messagebox.showerror("Camera initialization failed")
########################################################################################################################
#Camera source for single camera feed 
class CamSource:
    def __init__(self, camera_id, source_type, source_path, role='object', config=None):
        self.camera_id = camera_id
        self.source_type = source_type 
        self.source_path = source_path
        self.role = role
        self.config = config or {}
        self.is_active = False
        self.capture = None
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.last_frame = None
        self.fps_target = self.config.get('fps', 30 if role == 'priority' else 15)
        self.loop_video = self.config.get('loop', True)
############################################################################################################
#Camera manager --- multi-camera management
class CameraManager:    
    def __init__(self):
        self.cameras = {} 
        self.running = False
        self.frame_callbacks = {}
        
    def add_camera(self, camera_id, source_type, source_path, role='object', config=None):
        camera = CamSource(camera_id, source_type, source_path, role, config)
        self.cameras[camera_id] = camera
        self.frame_callbacks[camera_id] = []
        print(f"Added camera {camera_id}: {source_type} - {source_path} ({role})")
        return True
    
    def start_camera(self, camera_id):
        if camera_id not in self.cameras:
            return False
            
        camera = self.cameras[camera_id]
        if camera.is_active:
            return True

        if camera.source_type == 'usb':
            camera.capture = cv2.VideoCapture(int(camera.source_path))
        elif camera.source_type in ['rtsp', 'http']:
            camera.capture = cv2.VideoCapture(camera.source_path)
        elif camera.source_type == 'file':
            camera.capture = cv2.VideoCapture(camera.source_path)
        else:
            print(f"Unknown source type: {camera.source_type}")
            return False
        
        if not camera.capture.isOpened():
            print(f"Failed to open camera {camera_id}")
            return False

        self._configure_camera(camera)
        
        camera.is_active = True
        camera.thread = threading.Thread(target=self.capture_loop, args=(camera,))
        camera.thread.daemon = True
        camera.thread.start()
        
        print(f"Started camera {camera_id}")
        return True
    
    def stop_camera(self, camera_id):
        if camera_id not in self.cameras:
            return False
            
        camera = self.cameras[camera_id]
        camera.is_active = False
        
        if camera.thread:
            camera.thread.join(timeout=2.0)
            
        if camera.capture:
            camera.capture.release()
            camera.capture = None
            
        print(f"Stopped camera {camera_id}")
        return True
    
    def start_all_cameras(self):
        self.running = True
        success_count = 0
        for camera_id in self.cameras:
            if self.start_camera(camera_id):
                success_count += 1
        print(f"Started {success_count}/{len(self.cameras)} cameras")
        return success_count
    
    def stop_all_cameras(self):
        self.running = False
        for camera_id in list(self.cameras.keys()):
            self.stop_camera(camera_id)
    
    def get_latest_frame(self, camera_id):
        if camera_id in self.cameras:
            return self.cameras[camera_id].last_frame
        return None
    
    def get_all_latest_frames(self):
        frames = {}
        for camera_id, camera in self.cameras.items():
            if camera.is_active and camera.last_frame is not None:
                frames[camera_id] = camera.last_frame.copy()
        return frames
    
    def _configure_camera(self, camera):
        if not camera.capture:
            return

        if camera.source_type == 'usb':
            camera.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.config.get('width', 1280))
            camera.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.config.get('height', 720))
            camera.capture.set(cv2.CAP_PROP_FPS, camera.fps_target)
            camera.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if camera.capture.isOpened():
            actual_width = int(camera.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(camera.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = camera.capture.get(cv2.CAP_PROP_FPS)
            print(f"Camera {camera.camera_id} settings: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    def capture_loop(self, camera):
        frame_time = 1.0 / camera.fps_target
        last_time = time.time()
        
        while camera.is_active and self.running:
            ret, frame = camera.capture.read()
            
            if not ret:
                if camera.source_type == 'file' and camera.loop_video:
                    camera.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print(f"Camera {camera.camera_id} read failed")
                    break
            camera.last_frame = frame

            current_time = time.time()
            elapsed = current_time - last_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()
        
        camera.is_active = False
        print(f"Capture loop ended for camera {camera.camera_id}")

####################################################################################################################################
# save_detection_log 
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
    except Exception as e:
        print(f"Error writing to log file: {str(e)}")       
####################################################################################################################################
# cam_config to work with camera manager--- fallback
def cam_config():
    try:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise ValueError("Camera Error")
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  
        video_capture.set(cv2.CAP_PROP_FPS, 30)           
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        actual_fps = video_capture.get(cv2.CAP_PROP_FPS)
        actual_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Fallback camera settings: {actual_width}x{actual_height} @ {actual_fps} FPS")    
        
        return video_capture
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None

####################################################################################################################################
# clean exit 
def exit_app():
    global camera_manager
    if camera_manager:
        camera_manager.stop_all_cameras()
    cv2.destroyAllWindows()
    root.destroy()
####################################################################################################################################
#Screenshot of all cameras
def take_screenshot_all_cameras():
    try:
        all_frames = camera_manager.get_all_latest_frames()
        
        if not all_frames:
            messagebox.showerror("Screenshot failure", "No active cameras")
            return
        
        screenshot_dir = "screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        
        for camera_id, frame in all_frames.items():
            # Process frame for gun detection
            gun_results = gun_model(frame)
            for result in gun_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cls_id = int(box.cls[0])
                    label = f"{gun_model.names[cls_id]}: {conf:.2f}"

                    if conf > 0.76:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
            
            # Save screenshot for this camera
            filename = f"Eyespy_{camera_id}_{timestamp}.png"
            filepath = os.path.join(screenshot_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_files.append(filename)
        
        messagebox.showinfo("Eyespy+", f"Screenshots saved: {', '.join(saved_files)}")
        
    except Exception as e:
        messagebox.showerror("Screenshot Error", f"Error capturing screenshots: {str(e)}")

#screenshot single camera
def take_screenshot_detection_only():
    try:
        all_frames = camera_manager.get_all_latest_frames()
        
        if not all_frames:
            messagebox.showerror("Screenshot failure", "No active cameras")
            return
        
        screenshot_dir = "screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        detection_info = []
        
        for camera_id, frame in all_frames.items():
            gun_results = gun_model(frame)
            detections_found = False
            detection_count = 0
            
            for result in gun_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        cls_id = int(box.cls[0])
                        label = f"{gun_model.names[cls_id]}: {conf:.2f}"

                        if conf > 0.76:
                            detections_found = True
                            detection_count += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
            
            if detections_found:
                filename = f"Eyespy_{camera_id}_{timestamp}.png"
                filepath = os.path.join(screenshot_dir, filename)
                cv2.imwrite(filepath, frame)
                saved_files.append(filename)
                detection_info.append({
                    'camera_id': camera_id,
                    'filename': filename,
                    'detection_count': detection_count,
                    'timestamp': timestamp
                })
        
        if saved_files:
            messagebox.showinfo("Eyespy+", f"Detection screenshots saved: {', '.join(saved_files)}")
            print(f"Detection info: {detection_info}")  
            return detection_info  
        else:
            messagebox.showinfo("Eyespy+", "No detections found - no screenshots saved")
            return None
        
    except Exception as e:
        messagebox.showerror("Screenshot Error", f"Error capturing screenshots: {str(e)}")
        return None
    
#screenshot with detection frames
def take_screenshot_with_frames(detection_frames=None):
    try:
        #Use provided detection frames or fall back to current frames
        if detection_frames:
            frames_to_save = detection_frames
            source = "detection frames"
        else:
            frames_to_save = camera_manager.get_all_latest_frames()
            source = "current frames"
        
        if not frames_to_save:
            messagebox.showerror("Screenshot failure", "No frames available")
            return
        
        screenshot_dir = "screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        
        for camera_id, frame in frames_to_save.items():
            filename = f"Eyespy_{camera_id}_{timestamp}.png"
            filepath = os.path.join(screenshot_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_files.append(filename)
        
        print(f"Screenshots saved from {source}: {', '.join(saved_files)}")
        messagebox.showinfo("Eyespy+", f"Detection screenshots saved: {', '.join(saved_files)}")
        
        return saved_files
        
    except Exception as e:
        messagebox.showerror("Screenshot Error", f"Error capturing screenshots: {str(e)}")    
####################################################################################################################################
# nonblocking alert + play alarm sound
def show_nonblocking_alert(title, message, detection_frames=None):
    def show_alert():
        try:
            winsound.PlaySound('Sounds\\siren.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Sound error: {str(e)}")
        
        alert_window = tk.Toplevel(root)
        alert_window.title(title)
        alert_window.geometry("300x150")
        alert_window.configure(bg="red")  
        alert_window.attributes("-topmost", True)
        
        msg_label = tk.Label(alert_window, text=message, font=("Arial", 16), bg="red", fg="white", wraplength=280)
        msg_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        dismiss_button = tk.Button(alert_window, text="Dismiss", command=alert_window.destroy, font=("Arial", 12))
        dismiss_button.pack(pady=10)
        
        #Use detection frames if provided, otherwise fall back to current frames
        take_screenshot_with_frames(detection_frames)
        
        alert_window.after(5000, alert_window.destroy)
        
    alert_thread = threading.Thread(target=show_alert)
    alert_thread.daemon = True 
    alert_thread.start()

#alert flags to track per camera
gun_alert_states = {
    'priority_1': {'triggered': False, 'last_time': 0},
    'priority_2': {'triggered': False, 'last_time': 0},
    'object_1': {'triggered': False, 'last_time': 0},
    'object_2': {'triggered': False, 'last_time': 0},
    'object_3': {'triggered': False, 'last_time': 0}
}    
##########################################################################################################################################
#multi-camera gun detection processing
gun_detection_counter = 0

def process_gun_detections_multi_camera(camera_frames_dict):
    global gun_alert_states, gun_detection_counter
    
    # Process every 3rd frame to save processing power
    gun_detection_counter += 1
    if gun_detection_counter % 3 != 0:
        return camera_frames_dict
    
    processed_frames = {}
    current_time = time.time()
    detection_frames = {}  
    for camera_id, frame in camera_frames_dict.items():
        processed_frame = frame.copy()
        
        try:
            results = gun_model(frame, verbose=False) 
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                            x1, y1, x2, y2 = map(int, coords)
                            
                            conf = float(box.conf[0]) if isinstance(box.conf, (list, tuple, np.ndarray)) else float(box.conf)
                            cls_id = int(box.cls[0]) if isinstance(box.cls, (list, tuple, np.ndarray)) else int(box.cls)
                            label = f"{gun_model.names[cls_id]}: {conf:.2f}"
                            
                            if conf > 0.76:
                                print(f"Gun detection on {camera_id}: {label} at coordinates ({x1}, {y1}, {x2}, {y2})")
                                
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
                                save_detection_log(label, (x1, y1, x2-x1, y2-y1), f"Weapon Detection - {camera_id}")
                                
                                #Store frame with detection for screenshot
                                detection_frames[camera_id] = processed_frame.copy()
                                
                                # Per-camera alert logic
                                if gun_model.names[cls_id] == 'Gun' and conf > 0.76:
                                    camera_alert_state = gun_alert_states[camera_id]
                                    
                                    if not camera_alert_state['triggered'] or (current_time - camera_alert_state['last_time']) > 10.0:
                                        print(f"FIREARM ALERT TRIGGERED on {camera_id}!")
                                        
                                        # Pass the detection frame directly
                                        show_nonblocking_alert("EyeSpy ALERT!", f"FIREARM DETECTED on {camera_id.upper()}!", detection_frames)
                                        screenshot_payload(bot_token="########", chat_id="######", 
                                                         title=f"FIREARM ALERT - {camera_id}", 
                                                         message=f"FIREARM DETECTED on {camera_id}!")
                                        
                                        camera_alert_state['triggered'] = True
                                        camera_alert_state['last_time'] = current_time
                                        
                                        #Reset alert after cooldown
                                        def reset_gun_alert_flag(cam_id):
                                            gun_alert_states[cam_id]['triggered'] = False
                                        
                                        timer = threading.Timer(10.0, reset_gun_alert_flag, args=[camera_id])
                                        timer.daemon = True
                                        timer.start()
                        
                        except Exception as e:
                            print(f"Error processing gun detection box for {camera_id}: {str(e)}")
                            continue
            
        except Exception as e:
            print(f"Error in gun detection for {camera_id}: {str(e)}")
        
        processed_frames[camera_id] = processed_frame
    
    return processed_frames
#######################################################################################################################################################
# If camera is USB, flip it horizontally for mirror display

def should_flip_camera(actual_camera_id):
    if actual_camera_id in camera_manager.cameras:
        camera = camera_manager.cameras[actual_camera_id]
        return camera.source_type == 'usb'  #only flip USB cameras
    return False

def create_display_matrix(processed_frames):
    priority_width, priority_height = 400, 300  
    object_width, object_height = 266, 200     

    placeholder_priority = np.zeros((priority_height, priority_width, 3), dtype=np.uint8)
    placeholder_object = np.zeros((object_height, object_width, 3), dtype=np.uint8)
    
    cv2.putText(placeholder_priority, "No Signal", (priority_width//2-60, priority_height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    cv2.putText(placeholder_object, "No Signal", (object_width//2-50, object_height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    display_frames = {}

    for camera_id in ['priority_1', 'priority_2']:
        if camera_id in processed_frames:
            frame = processed_frames[camera_id]
            if should_flip_camera(camera_id):
                frame = cv2.flip(frame, 1)
            display_frames[camera_id] = cv2.resize(frame, (priority_width, priority_height))

            cv2.putText(display_frames[camera_id], f"{camera_id.upper()}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display_frames[camera_id] = placeholder_priority.copy()
            cv2.putText(display_frames[camera_id], f"{camera_id.upper()}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Process object cameras
    for camera_id in ['object_1', 'object_2', 'object_3']:
        if camera_id in processed_frames:
            frame = processed_frames[camera_id]
            if should_flip_camera(camera_id):
                frame = cv2.flip(frame, 1)
            display_frames[camera_id] = cv2.resize(frame, (object_width, object_height))
            cv2.putText(display_frames[camera_id], f"{camera_id.upper()}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            display_frames[camera_id] = placeholder_object.copy()
            cv2.putText(display_frames[camera_id], f"{camera_id.upper()}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    #Create matrix layout
    try:
        top_row = cv2.hconcat([display_frames['priority_1'], display_frames['priority_2']])

        bottom_row = cv2.hconcat([display_frames['object_1'], display_frames['object_2'], display_frames['object_3']])

        top_width = top_row.shape[1]
        bottom_width = bottom_row.shape[1]
        
        if top_width != bottom_width:
            bottom_row = cv2.resize(bottom_row, (top_width, bottom_row.shape[0]))
        
        matrix_display = cv2.vconcat([top_row, bottom_row])
        
        return matrix_display
        
    except Exception as e:
        print(f"Error creating display matrix: {e}")
        error_frame = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Display Matrix Error", (250, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return error_frame

def update_frame():
    try:
        all_frames = camera_manager.get_all_latest_frames()
        
        if not all_frames:
            no_signal_frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(no_signal_frame, "No Active Cameras", (250, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)
            cv2.putText(no_signal_frame, "Click 'Start Cameras' to begin", (220, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            
            cv2_img = cv2.cvtColor(no_signal_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_img)
            imgtk = ImageTk.PhotoImage(image=img)
            matrix_display_label.configure(image=imgtk)
            matrix_display_label.image = imgtk
            
            root.after(100, update_frame)
            return
        
        # Process gun detections on all frames
        processed_frames = process_gun_detections_multi_camera(all_frames)
        
        # Create matrix display from processed frames
        matrix_frame = create_display_matrix(processed_frames)
        
        #Convert matrix to display format
        cv2_img = cv2.cvtColor(matrix_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_img)
        imgtk = ImageTk.PhotoImage(image=img)

        matrix_display_label.configure(image=imgtk)
        matrix_display_label.image = imgtk 
        
        update_camera_status()

        root.after(50, update_frame)  

    except Exception as e:
        messagebox.showerror("Frame Update Error", f"Error during frame update: {str(e)}")
        root.after(100, update_frame) 

def update_camera_status():
    active_cameras = [cam_id for cam_id, cam in camera_manager.cameras.items() if cam.is_active]
    total_cameras = len(camera_manager.cameras)
    
    # Create detailed status
    status_parts = []
    for camera_id, camera in camera_manager.cameras.items():
        if camera.is_active:
            status_parts.append(f"{camera_id}:✓")
        else:
            status_parts.append(f"{camera_id}:✗")
    
    status_text = f"Cameras: {len(active_cameras)}/{total_cameras} active | {' | '.join(status_parts)}"
    status_label.config(text=status_text)

################################################################################################################################
# notification to Telegram with latest screenshot - UNCHANGED from your original code

def screenshot_payload(bot_token, chat_id, title, message):
  # Use latest screenshot
  files = glob.glob("screenshots/Eyespy_capture_*.png")
  if not files:
      files = glob.glob("screenshots/Eyespy_*.png")  
  
  if files:
      latest_file = max(files, key=os.path.getmtime)
      url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
      caption = f"{title}\n\n{message}"
     
      try:
          with open(latest_file, 'rb') as photo:
            response = requests.post(url, files={'photo': photo}, data={'chat_id': chat_id, 'caption': caption})
          # verify response + error handling
          if response.status_code == 200:
            print("Sent to Telegram Server")
          else:
            print(f"Failed: {response.text}")
      except Exception as e:
          print(f"Telegram error: {e}")

################################################################################################################################
# CHANGE: Completely modified main window layout for matrix display
root = tk.Tk()
root.title("Eyespy+ Multi-Camera Matrix View (Gun Detection Testing)")
root.geometry("1000x700") 

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

display_frame = tk.Frame(main_frame, relief='sunken', bd=2)
display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

matrix_display_label = tk.Label(display_frame, 
                                text="Multi-Camera Matrix View\n\nTop Row: Priority Cameras (Action Recognition Ready)\nBottom Row: Object Detection Cameras\n\nClick 'Start Cameras' to begin", 
                                width=80, height=30, bg='lightgray', relief='flat',
                                font=('Arial', 12), justify='center')
matrix_display_label.pack(fill=tk.BOTH, expand=True)

######################################################################################

button_frame = tk.Frame(main_frame)
button_frame.pack(fill=tk.X, pady=10)

info_frame = tk.Frame(button_frame)
info_frame.pack(fill=tk.X, pady=(0, 5))

info_label = tk.Label(info_frame, 
                     text="Matrix Layout: Top Row = Priority Cameras (400x300) | Bottom Row = Object Cameras (266x200)",
                     font=('Arial', 9), fg='blue')
info_label.pack()


controls_frame = tk.Frame(button_frame)
controls_frame.pack(fill=tk.X)

screenshot_button = ttk.Button(controls_frame, text="Screenshot All Cameras", command=take_screenshot_all_cameras)
screenshot_button.pack(side=tk.LEFT, padx=(0, 5))

def start_cameras():
    Camera_Source_Selection(root, camera_manager)

def stop_cameras():
    camera_manager.stop_all_cameras()

start_cameras_button = ttk.Button(controls_frame, text="Start Cameras", command=start_cameras)
start_cameras_button.pack(side=tk.LEFT, padx=(0, 5))

stop_cameras_button = ttk.Button(controls_frame, text="Stop Cameras", command=stop_cameras)
stop_cameras_button.pack(side=tk.LEFT, padx=(0, 5))

exit_button = ttk.Button(controls_frame, text="Exit", command=exit_app)
exit_button.pack(side=tk.RIGHT)

status_frame = tk.Frame(main_frame)
status_frame.pack(fill=tk.X, pady=(5, 0))

status_label = tk.Label(status_frame, text="Ready - No cameras active", relief='sunken', anchor='w')
status_label.pack(fill=tk.X)

camera_manager = CameraManager()

root.after(1000, update_frame) 

root.protocol("WM_DELETE_WINDOW", exit_app)
root.mainloop()
