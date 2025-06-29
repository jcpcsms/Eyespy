import cv2
import threading
import time
import numpy as np
from .camera_source import CamSource

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

            # Frame rate limiting
            current_time = time.time()
            elapsed = current_time - last_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()
        
        camera.is_active = False
        print(f"Capture loop ended for camera {camera.camera_id}")
