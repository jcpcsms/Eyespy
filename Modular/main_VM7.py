# EyeSpy+ Jetson-Integrated HTTP Camera Alert Bridge
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import time
import threading
import os
import winsound
import concurrent.futures
from threading import Lock
import queue
import requests
import re
from datetime import datetime
from typing import Dict, List, Optional

from detection.gun_detector import GunDetector
from detection.action_recognizer import ActionRecognition
from camera.camera_manager import CameraManager
from camera.camera_selector import Camera_Source_Selection
from gui.matrix_display import create_display_matrix
from gui.styles import DarkTheme
from utils.detection_logger import save_detection_log
from utils.telegram_notifier import screenshot_payload

######################################################################################################

class JetsonAlertBridge:
    def __init__(self, pc_alert_callback, telegram_callback):
        self.pc_alert_callback = pc_alert_callback
        self.telegram_callback = telegram_callback
        self.jetson_cameras = {}
        self.polling_active = False
        self.polling_thread = None
        self.poll_interval = 2.0
        self.alert_cooldown = {}
        print("Jetson Alert Bridge initialized - ready to monitor Jetson cameras")
    def add_jetson_camera(self, camera_id: str, jetson_ip: str, port: int): #build test measure 
        self.jetson_cameras[camera_id] = {
            'ip': jetson_ip,
            'port': port,
            'events_url': f'http://{jetson_ip}:{port}/events'
        }
        self.alert_cooldown[camera_id] = {'gun': 0, 'action': 0}
        print(f"Added Jetson camera {camera_id} at {jetson_ip}:{port}")
    
    def start_monitoring(self):
        if self.polling_active:
            return
        
        self.polling_active = True
        self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.polling_thread.start()
        print(f"Alert monitoring started for {len(self.jetson_cameras)} Jetson cameras")
    
    def stop_monitoring(self):
        self.polling_active = False
        if self.polling_thread:
            self.polling_thread.join(timeout=5)
        print("Alert monitoring stopped")
    
    def _polling_loop(self):
        while self.polling_active:
            try:
                for camera_id in list(self.jetson_cameras.keys()):
                    self._check_camera_alerts(camera_id)
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"Alert polling error: {e}")
                time.sleep(5)
    
    def _check_camera_alerts(self, camera_id: str):
        camera_info = self.jetson_cameras.get(camera_id)
        if not camera_info:
            return
        
        try:
            response = requests.get(camera_info['events_url'], timeout=3)
            if response.status_code != 200:
                return
            
            events_data = response.json()
            current_time = time.time()
            
            
            gun_detections = events_data.get('gun_detections', [])
            for detection in gun_detections[-5:]:  
                confidence = detection.get('confidence', 0)
                timestamp = detection.get('timestamp', 0)
                gun_class = detection.get('class', '')
                
                if (confidence > 0.65 and 
                    gun_class == 'Gun' and 
                    timestamp > self.alert_cooldown[camera_id]['gun']):
                    
                    self._trigger_gun_alert(camera_id, detection)
                    self.alert_cooldown[camera_id]['gun'] = current_time + 10.0
                    break
            
            # Process SOS alerts
            action_detections = events_data.get('action_detections', [])
            for detection in action_detections[-5:]:
                action = detection.get('action', '')
                confidence = detection.get('confidence', 0)
                timestamp = detection.get('timestamp', 0)
                
                if (action == 'circular_wave' and 
                    confidence > 0.80 and 
                    timestamp > self.alert_cooldown[camera_id]['action']):
                    
                    self._trigger_sos_alert(camera_id, detection)
                    self.alert_cooldown[camera_id]['action'] = current_time + 10.0
                    break
                    
        except requests.RequestException:
            pass 
        except Exception as e:
            print(f"Error checking {camera_id} alerts: {e}")
    
    def _trigger_gun_alert(self, camera_id: str, detection: Dict):
        try:
            confidence = detection.get('confidence', 0)
            title = "FIREARM DETECTED!"
            message = f"FIREARM on {camera_id.replace('_', ' ').upper()} (Jetson AI)!\nConfidence: {confidence:.1%}"
            
            if not mute_notifications:
                
                threading.Thread(
                    target=self.pc_alert_callback,
                    args=(title, message, None),
                    daemon=True
                ).start()
                
                # Telegram notification
                self.telegram_callback(
                    bot_token="#####s",
                    chat_id="#####",
                    title="JETSON FIREARM ALERT",
                    message=f"FIREARM DETECTED on {camera_id} (Jetson AI)"
                )
                print(f"Jetson Gun alert triggered for {camera_id}")
            else:
                print(f"MUTED JETSON ALERT: FIREARM on {camera_id}")
                
        except Exception as e:
            print(f"Error triggering gun alert: {e}")
    
    def _trigger_sos_alert(self, camera_id: str, detection: Dict):
        try:
            confidence = detection.get('confidence', 0)
            title = "SOS DETECTED!"
            message = f"SOS on {camera_id.replace('_', ' ').upper()} (Jetson AI)!\nConfidence: {confidence:.1%}"

            if not mute_notifications:
    
                threading.Thread(
                    target=self.pc_alert_callback,
                    args=(title, message, None),
                    daemon=True
                ).start()
                
                # Telegram notification
                self.telegram_callback(
                    bot_token="#####s",
                    chat_id="#####",
                    title="JETSON SOS ALERT", 
                    message=f"SOS DETECTED on {camera_id} (Jetson AI)"
                )
                print(f"Jetson SOS alert triggered for {camera_id}")
            else:
                print(f"MUTED JETSON ALERT: SOS on {camera_id}")
                
        except Exception as e:
            print(f"Error triggering SOS alert: {e}")

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
gun_detection_queue = queue.Queue()
action_recognition_queue = queue.Queue()
gun_detection_frame_counter = 0
GUN_DETECTION_FRAME_INTERVAL = 20
detection_results = {}
detection_results_lock = Lock()

gun_detector = GunDetector()
action_recognizers = {}
camera_manager = CameraManager()
jetson_alert_bridge = None

mute_notifications = False

gun_alert_states = {
    'priority_1': {'triggered': False, 'last_time': 0},
    'priority_2': {'triggered': False, 'last_time': 0},
    'object_1': {'triggered': False, 'last_time': 0},
    'object_2': {'triggered': False, 'last_time': 0},
    'object_3': {'triggered': False, 'last_time': 0}
}

circular_wave_alert_states = {}

def is_jetson_camera_source(camera):
    if not camera or camera.source_type != 'http':
        return False
    
    jetson_ips = ['192.168.100.2', '192.168.1.181']
    source_path = camera.source_path.lower()
    
    for jetson_ip in jetson_ips:
        if jetson_ip in source_path:
            return True
    
    return False

def auto_detect_jetson_cameras(camera_manager) -> List[Dict]:
    jetson_cameras = []
    
    for camera_id, camera in camera_manager.cameras.items():
        if camera.source_type == 'http':
            try:
                source_path = camera.source_path.lower()
                jetson_ips = ['192.168.100.2', '192.168.1.181']
                
                for jetson_ip in jetson_ips:
                    if jetson_ip in source_path:
                        port_match = re.search(rf'{jetson_ip}:(\d+)', source_path)
                        port = int(port_match.group(1)) if port_match else 5000
                        
                        jetson_cameras.append({
                            'camera_id': camera_id,
                            'jetson_ip': jetson_ip,
                            'port': port,
                            'role': camera.role
                        })
                        break
                        
            except Exception as e:
                print(f"Error parsing Jetson camera {camera_id}: {e}")
    
    return jetson_cameras

def setup_jetson_alert_bridge():
    global jetson_alert_bridge
    
    jetson_alert_bridge = JetsonAlertBridge(show_nonblocking_alert, screenshot_payload)
    
    jetson_cameras = auto_detect_jetson_cameras(camera_manager)
    
    if jetson_cameras:
        print(f"Setting up alert bridge for {len(jetson_cameras)} Jetson cameras")
        
        for cam_info in jetson_cameras:
            jetson_alert_bridge.add_jetson_camera(
                cam_info['camera_id'],
                cam_info['jetson_ip'], 
                cam_info['port']
            )
        
        jetson_alert_bridge.start_monitoring()
        print("Jetson alert bridge active - PC will receive Jetson alerts!")
        return True
    else:
        print("No Jetson cameras detected for alert bridge")
        return False

def cleanup_jetson_alert_bridge():
    global jetson_alert_bridge
    if jetson_alert_bridge:
        jetson_alert_bridge.stop_monitoring()

def take_screenshot_with_frames(detection_frames=None):
    try:
        frames_to_save = detection_frames if detection_frames else camera_manager.get_all_latest_frames()
        
        if not frames_to_save:
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
        
        return saved_files
        
    except Exception as e:
        print(f"Screenshot error: {e}")

def show_nonblocking_alert(title, message, detection_frames=None):
    print(f"Alert function called: {title}")
    
    def show_alert():
        print(f"Alert thread started: {title}")
        
        try:
            winsound.PlaySound('sounds\\siren.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Sound error: {e}")
        
        try:
            alert_window = tk.Toplevel(root)
            alert_window.title(title)
            alert_window.geometry("350x180")
            alert_window.configure(bg=DarkTheme.ERROR)
            alert_window.attributes("-topmost", True)
            print("Alert window created")
            
            msg_label = tk.Label(alert_window, text=message, font=("Arial", 14, "bold"), 
                               bg=DarkTheme.ERROR, fg=DarkTheme.FG_PRIMARY, wraplength=320)
            msg_label.pack(expand=True, fill=tk.BOTH, padx=15, pady=15)
            print("Message label added")
            
            dismiss_button = tk.Button(alert_window, text="DISMISS", 
                                     command=alert_window.destroy, 
                                     font=("Arial", 12, "bold"),
                                     bg=DarkTheme.BG_PRIMARY, fg=DarkTheme.FG_PRIMARY,
                                     activebackground=DarkTheme.BG_SECONDARY,
                                     bd=0, padx=20, pady=10)
            dismiss_button.pack(pady=(0, 15))
            print("Dismiss button added")
            
            try:
                take_screenshot_with_frames(detection_frames)
                print("Screenshot taken")
            except Exception as e:
                print(f"Screenshot error: {e}")
            
            alert_window.after(5000, alert_window.destroy)
            print("Auto-close timer set")
            
        except Exception as e:
            print(f"Window creation error: {e}")
            import traceback
            traceback.print_exc()
    
    alert_thread = threading.Thread(target=show_alert)
    alert_thread.daemon = True
    alert_thread.start()
    print(f"Alert thread started for: {title}")

def resize_frame_for_purpose(frame, purpose='display'):
    height, width = frame.shape[:2]
    
    resolutions = {
        'display': (320, 240),
        'detection': (960, 720),
        'action': (640, 480),
        'screenshot': (1280, 720)
    }
    
    max_width, max_height = resolutions.get(purpose, (640, 480))
    
    if width > max_width or height > max_height:
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    return frame

def process_gun_detection_threaded(camera_id, frame):
    processed_frame, alert_needed = gun_detector.process_gun_detections(frame)
    return camera_id, processed_frame, alert_needed

def process_multi_camera_detections(camera_frames_dict):
    global gun_alert_states, circular_wave_alert_states, gun_detection_frame_counter
    
    current_time = time.time()
    
    display_frames = {}
    detection_frames = {}
    original_frames = {}
    
    for camera_id, frame in camera_frames_dict.items():
        original_frames[camera_id] = frame.copy()
        display_frames[camera_id] = resize_frame_for_purpose(frame.copy(), 'display')
        detection_frames[camera_id] = resize_frame_for_purpose(frame.copy(), 'detection')
    
    gun_detection_frame_counter += 1
    should_detect_guns = (gun_detection_frame_counter % GUN_DETECTION_FRAME_INTERVAL == 0)
    
    gun_futures = []
    if should_detect_guns:
        for camera_id in detection_frames:
            camera = camera_manager.cameras.get(camera_id)

            if is_jetson_camera_source(camera):
                continue

            future = executor.submit(process_gun_detection_threaded, camera_id, detection_frames[camera_id])
            gun_futures.append(future)
    
    if gun_futures:
        try:
            for future in concurrent.futures.as_completed(gun_futures, timeout=0.5):
                try:
                    camera_id, processed_frame, alert_needed = future.result()
                    display_frames[camera_id] = resize_frame_for_purpose(processed_frame, 'display')
                    
                    if alert_needed:
                        camera_alert_state = gun_alert_states[camera_id]
                        
                        if not camera_alert_state['triggered'] or (current_time - camera_alert_state['last_time']) > 10.0:
                            screenshot_frame = resize_frame_for_purpose(original_frames[camera_id], 'screenshot')

                            if mute_notifications:
                                print(f"MUTED ALERT: FIREARM on {camera_id.replace('_', ' ').upper()}!")
                            else:
                                threading.Thread(
                                    target=show_nonblocking_alert,
                                    args=("FIREARM DETECTED!", 
                                        f"FIREARM on {camera_id.replace('_', ' ').upper()}!",
                                        {camera_id: screenshot_frame})
                                ).start()
                                screenshot_payload(bot_token="####s", chat_id="#####", title="FIREARM ALERT", message="FIREARM DETECTED!")
                            
                            camera_alert_state['triggered'] = True
                            camera_alert_state['last_time'] = current_time
                            
                            timer = threading.Timer(10.0, lambda: gun_alert_states[camera_id].update({'triggered': False}))
                            timer.daemon = True
                            timer.start()
                            
                except Exception as e:
                    print(f"Gun detection error: {e}")
        except concurrent.futures.TimeoutError:
            print("Gun detection futures timed out")
    
    return display_frames

def update_camera_status():
    active_cameras = [cam_id for cam_id, cam in camera_manager.cameras.items() if cam.is_active]
    total_cameras = len(camera_manager.cameras)
    
    jetson_count = 0
    local_count = 0
    
    status_parts = []
    for camera_id, camera in camera_manager.cameras.items():
        if camera.is_active:
            if is_jetson_camera_source(camera):
                status_parts.append(f"{camera_id}:OK(J)")
                jetson_count += 1
            else:
                status_parts.append(f"{camera_id}:OK")
                local_count += 1
        else:
            status_parts.append(f"{camera_id}:OFF")
    
    bridge_status = "Bridge: ON" if jetson_alert_bridge and jetson_alert_bridge.polling_active else "Bridge: OFF"
    notifications_status = "MUTED" if mute_notifications else "ACTIVE"
    
    status_text = f"Cameras: {len(active_cameras)}/{total_cameras} active | Local: {local_count} | Jetson: {jetson_count} | {bridge_status} | Notifications: {notifications_status}"
    status_label.config(text=status_text)

def update_frame():
    try:
        all_frames = camera_manager.get_all_latest_frames()
        
        if not all_frames:
            no_signal_frame = np.full((600, 800, 3), 30, dtype=np.uint8)
            cv2.putText(no_signal_frame, "No Active Cameras", (200, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150, 150, 150), 3)
            cv2.putText(no_signal_frame, "Click 'Start Cameras' to begin", (180, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            
            cv2_img = cv2.cvtColor(no_signal_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_img)
            imgtk = ImageTk.PhotoImage(image=img)
            matrix_display_label.configure(image=imgtk)
            matrix_display_label.image = imgtk
            
            root.after(100, update_frame)
            return
        
        processed_frames = process_multi_camera_detections(all_frames)
        
        matrix_frame = create_display_matrix(processed_frames, camera_manager)
        
        cv2_img = cv2.cvtColor(matrix_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_img)
        imgtk = ImageTk.PhotoImage(image=img)
        matrix_display_label.configure(image=imgtk)
        matrix_display_label.image = imgtk
        
        update_camera_status()
        root.after(50, update_frame)
        
    except Exception as e:
        print(f"Frame update error: {e}")
        root.after(100, update_frame)

def take_screenshot_all():
    try:
        all_frames = camera_manager.get_all_latest_frames()
        if all_frames:
            saved = take_screenshot_with_frames(all_frames)
            if saved:
                messagebox.showinfo("Screenshots Saved", f"Saved {len(saved)} screenshots")
        else:
            messagebox.showwarning("No Cameras", "No active cameras to screenshot")
    except Exception as e:
        messagebox.showerror("Error", f"Screenshot error: {str(e)}")

def exit_app():
    cleanup_jetson_alert_bridge()
    camera_manager.stop_all_cameras()
    executor.shutdown(wait=True)
    cv2.destroyAllWindows()
    root.destroy()

root = tk.Tk()
root.title("EyeSpy+ Milestone 7 - Jetson USB Camera Streaming")
root.geometry("1100x750")
root.configure(bg=DarkTheme.BG_PRIMARY)

DarkTheme.configure_styles(root)

main_frame = tk.Frame(root, bg=DarkTheme.BG_PRIMARY)
main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

title_label = tk.Label(main_frame, text="EyeSpy+ Milestone 7 - Jetson USB Camera Streaming",
                      font=("Arial", 18, "bold"), bg=DarkTheme.BG_PRIMARY, 
                      fg=DarkTheme.FG_PRIMARY)
title_label.pack(pady=(0, 10))

display_frame = tk.Frame(main_frame, bg=DarkTheme.BG_SECONDARY, 
                        highlightbackground=DarkTheme.BG_TERTIARY, 
                        highlightthickness=2)
display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

matrix_display_label = tk.Label(display_frame, bg=DarkTheme.BG_SECONDARY)
matrix_display_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

info_frame = tk.Frame(main_frame, bg=DarkTheme.BG_PRIMARY)
info_frame.pack(fill=tk.X, pady=(0, 10))

info_label = tk.Label(info_frame, 
                     text="Jetson Cameras: Full CV Processing + PC Alerts | Local Cameras: Gun Detection Only | (J) = Jetson ",
                     font=('Arial', 10), bg=DarkTheme.BG_PRIMARY, fg=DarkTheme.FG_SECONDARY)
info_label.pack()

controls_frame = tk.Frame(main_frame, bg=DarkTheme.BG_PRIMARY)
controls_frame.pack(fill=tk.X, pady=(0, 10))

button_style = {
    'font': ('Arial', 11),
    'bg': DarkTheme.BG_TERTIARY,
    'fg': DarkTheme.FG_PRIMARY,
    'activebackground': DarkTheme.BG_SECONDARY,
    'activeforeground': DarkTheme.FG_PRIMARY,
    'bd': 0,
    'padx': 15,
    'pady': 8,
    'cursor': 'hand2'
}

def toggle_mute_notifications():
    global mute_notifications
    mute_notifications = not mute_notifications
    status = "MUTED" if mute_notifications else "ACTIVE"
    print(f"Notifications {status}")
    
    # Update button text
    mute_btn.config(
        text=f"Unmute" if mute_notifications else "Mute",
        bg=DarkTheme.WARNING if mute_notifications else DarkTheme.BG_TERTIARY
    )

def start_cameras():
    Camera_Source_Selection(root, camera_manager)
    
    def setup_bridge():
        time.sleep(2)  # Wait for cameras to start
        setup_jetson_alert_bridge()
    
    threading.Thread(target=setup_bridge, daemon=True).start()

start_btn = tk.Button(controls_frame, text="Start Cameras", 
                     command=start_cameras, **button_style)
start_btn.pack(side=tk.LEFT, padx=(0, 10))

stop_btn = tk.Button(controls_frame, text="Stop Cameras", 
                    command=camera_manager.stop_all_cameras, **button_style)
stop_btn.pack(side=tk.LEFT, padx=(0, 10))

screenshot_btn = tk.Button(controls_frame, text="Screenshot All", 
                          command=take_screenshot_all, **button_style)
screenshot_btn.pack(side=tk.LEFT, padx=(0, 10))

mute_btn = tk.Button(controls_frame, text="Mute", 
                    command=toggle_mute_notifications, **button_style)
mute_btn.pack(side=tk.LEFT, padx=(0, 10))

exit_btn = tk.Button(controls_frame, text="Exit", command=exit_app,
                    font=('Arial', 11, 'bold'), bg=DarkTheme.ERROR,
                    fg=DarkTheme.FG_PRIMARY, activebackground='#c0392b',
                    activeforeground=DarkTheme.FG_PRIMARY, bd=0, 
                    padx=20, pady=8, cursor='hand2')
exit_btn.pack(side=tk.RIGHT)

status_frame = tk.Frame(root, bg=DarkTheme.BG_SECONDARY)
status_frame.pack(fill=tk.X, side=tk.BOTTOM)

status_label = tk.Label(status_frame, text="Ready - Jetson Integration + Alert Bridge Active", 
                       bg=DarkTheme.BG_SECONDARY, fg=DarkTheme.FG_SECONDARY,
                       anchor='w', padx=10, pady=5)
status_label.pack(fill=tk.X)

root.after(1000, update_frame)
root.protocol("WM_DELETE_WINDOW", exit_app)

if __name__ == "__main__":
    root.mainloop()
