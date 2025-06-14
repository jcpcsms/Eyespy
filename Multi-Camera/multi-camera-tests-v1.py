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
########################################################################################################
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

###########################################################################################
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

        title_label = ttk.Label(main_frame, text="Configure Camera Sources", font=("Ebrima", 14))
        title_label.pack(pady=(0, 10))
        
        self.create_camera_config('P1', 'Priority Camera 1', main_frame)
        self.create_camera_config('P2', 'Priority Camera 2', main_frame)
        self.create_camera_config('cam1', 'Camera 1', main_frame)
        self.create_camera_config('cam2', 'Camera 2', main_frame)
        self.create_camera_config('cam3', 'Camera 3', main_frame)
        
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
        type_combo = ttk.Combobox(frame, 
                                  textvariable=self.camera_configs[camera_id]['source_type'], 
                                 values=['usb', 'file', 'rtsp', 'http'],
                                   width=8, 
                                   state='readonly')
        type_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame, text="Source:").pack(side=tk.LEFT, padx=(10, 0))
        source_entry = ttk.Entry(frame, textvariable=self.camera_configs[camera_id]['source_path'], width=30)
        source_entry.pack(side=tk.LEFT, padx=5)

        # Browse button for video files
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
        # Stop all existing cameras before starting new ones
        self.camera_manager.stop_all_cameras()
        self.camera_manager.cameras.clear()

        # Now add selected cameras
        for camera_id, config in self.camera_configs.items():
            if config['enabled'].get():
                source_type = config['source_type'].get()
                source_path = config['source_path'].get()
                
                if source_path.strip(): 
                    role = 'priority' if camera_id.startswith('P') else 'cam'
                    # Loop video files like the Italian Job ;)
                    camera_config = {'loop': True} if source_type == 'file' else {}
                    self.camera_manager.add_camera(camera_id, source_type, source_path, role, camera_config)

        #start video feed from camera sources
        if not self.camera_manager.cameras:
            messagebox.showwarning("No Cameras Selected", "Please select at least one camera source to start.")
            return
        success_count = self.camera_manager.start_all_cameras()
        
        if success_count > 0:
            messagebox.showinfo("Success", f"Started {success_count} cameras")
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", "No cameras could be started. Check your configurations.")
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
        # Make Priority Cameras show at 30 FPS, others at 15 FPS
        self.fps_target = self.config.get('fps', 30 if role == 'priority' else 15)
        self.loop_video = self.config.get('loop', True)
############################################################################################################
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
            print(f"Invalid source: {camera.source_type}")
            return False
        
        if not camera.capture.isOpened():
            print(f"Camera failed {camera_id}")
            return False

        self.config_camera(camera)

        camera.is_active = True
        camera.thread = threading.Thread(target=self._capture_loop, args=(camera,))
        camera.thread.daemon = True
        camera.thread.start()
        
        print(f"Started camera {camera_id}")
        return True
    
    def stop_camera(self, camera_id):
        """Stop a specific camera"""
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
        """Start all added cameras"""
        self.running = True
        success_count = 0
        for camera_id in self.cameras:
            if self.start_camera(camera_id):
                success_count += 1
        print(f"Started {success_count}/{len(self.cameras)} cameras")
        return success_count
    
    def stop_all_cameras(self):
        """Stop all cameras"""
        self.running = False
        for camera_id in list(self.cameras.keys()):
            self.stop_camera(camera_id)
    
    def get_latest_frame(self, camera_id):
        """Get the most recent frame from a camera"""
        if camera_id in self.cameras:
            return self.cameras[camera_id].last_frame
        return None
    
    def get_all_latest_frames(self):
        """Get latest frames from all active cameras"""
        frames = {}
        for camera_id, camera in self.cameras.items():
            if camera.is_active and camera.last_frame is not None:
                frames[camera_id] = camera.last_frame.copy()
        return frames
    
    def config_camera(self, camera):
        if not camera.capture:
            return
            
        if camera.source_type == 'usb':
            camera.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera.config.get('width', 1280))
            camera.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.config.get('height', 720))
            camera.capture.set(cv2.CAP_PROP_FPS, camera.fps_target)
            camera.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Log actual settings
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
                    # loop video file --- italian job style
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
        print(f"loop ended for camera {camera.camera_id}")
