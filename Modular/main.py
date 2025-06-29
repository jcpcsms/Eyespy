# EyeSpy+ Refactored Main Application
# Multi-camera Gun Detection and Action Recognition System
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
from detection.gun_detector import GunDetector
from detection.action_recognizer import ActionRecognition
from camera.camera_manager import CameraManager
from camera.camera_selector import Camera_Source_Selection
from gui.matrix_display import create_display_matrix
from gui.styles import DarkTheme
from utils.detection_logger import save_detection_log
from utils.telegram_notifier import screenshot_payload

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
gun_detection_queue = queue.Queue()
action_recognition_queue = queue.Queue()
gun_detection_frame_counter = 0
# Reduced interval to process gun detections every 20 frames
# This helps balance performance and detection frequency
GUN_DETECTION_FRAME_INTERVAL = 20
detection_results = {}
detection_results_lock = Lock()

gun_detector = GunDetector()
action_recognizers = {}
camera_manager = CameraManager()



gun_alert_states = {
    'priority_1': {'triggered': False, 'last_time': 0},
    'priority_2': {'triggered': False, 'last_time': 0},
    'object_1': {'triggered': False, 'last_time': 0},
    'object_2': {'triggered': False, 'last_time': 0},
    'object_3': {'triggered': False, 'last_time': 0}
}


circular_wave_alert_states = {}

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
            print("Sound played successfully")
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
        'display': (320, 240), #Reduced resolution for display - saves resources on PC
        'detection': (960, 720), #Gun Model sees this resolution
        'action': (640, 480), #Action recognition model sees this resolution
        'screenshot': (1280, 720) # Higher resolution for screenshots
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

def process_action_recognition_threaded(camera_id, frame, recognizer):
    frame = cv2.flip(frame, 1)
    
    action, confidence = recognizer.process_frame(frame)
    return camera_id, action, confidence

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
            future = executor.submit(process_gun_detection_threaded, camera_id, detection_frames[camera_id])
            gun_futures.append(future)
    
    action_futures = []
    for camera_id in display_frames:
        camera = camera_manager.cameras.get(camera_id)
        
        if camera and camera.role == 'priority':
            if camera_id not in action_recognizers:
                action_recognizers[camera_id] = ActionRecognition(use_jetson=True, jetson_ip='192.168.100.2')
                action_recognizers[camera_id].set_camera_id(camera_id)
                print(f"Created action recognizer for {camera_id}")
            
            action_frame = resize_frame_for_purpose(original_frames[camera_id], 'action')
            future = executor.submit(
                process_action_recognition_threaded,
                camera_id,
                action_frame,
                action_recognizers[camera_id]
            )
            action_futures.append(future)
    
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
                            
                            threading.Thread(
                                target=show_nonblocking_alert,
                                args=("FIREARM DETECTED!", 
                                    f"FIREARM on {camera_id.replace('_', ' ').upper()}!",
                                    {camera_id: screenshot_frame})
                            ).start()
                            screenshot_payload(bot_token="#######", chat_id="#######", title="FIREARM ALERT", message="FIREARM DETECTED!")
                            camera_alert_state['triggered'] = True
                            camera_alert_state['last_time'] = current_time
                            
                            timer = threading.Timer(10.0, lambda: gun_alert_states[camera_id].update({'triggered': False}))
                            timer.daemon = True
                            timer.start()
                            
                except Exception as e:
                    print(f"Gun detection error: {e}")
        except concurrent.futures.TimeoutError:
            print("Gun detection futures timed out")
    
    if action_futures:
        try:
            for future in concurrent.futures.as_completed(action_futures, timeout=1.0):
                try:
                    camera_id, action, confidence = future.result()
                    
                    if action == 'circular_wave' and confidence > 0.80:
                        if camera_id not in circular_wave_alert_states:
                            circular_wave_alert_states[camera_id] = {
                                'triggered': False,
                                'last_time': 0,
                                'detection_count': 0,
                                'first_detection_time': 0
                            }
                        
                        alert_state = circular_wave_alert_states[camera_id]
                        
                        if alert_state['detection_count'] == 0:
                            alert_state['first_detection_time'] = current_time
                        
                        alert_state['detection_count'] += 1
                        
                        if alert_state['detection_count'] >= 2:
                            if not alert_state['triggered'] or (current_time - alert_state['last_time']) > 10.0:
                                screenshot_frame = resize_frame_for_purpose(original_frames[camera_id], 'screenshot')
                                
                                threading.Thread(
                                    target=show_nonblocking_alert,
                                    args=("SOS DETECTED!",
                                        f"SOS on {camera_id.replace('_', ' ').upper()}!",
                                        {camera_id: screenshot_frame})
                                ).start()
                                screenshot_payload(bot_token="######", chat_id="####", title="SOS", message="SOS DETECTED!")
                                alert_state['triggered'] = True
                                alert_state['last_time'] = current_time
                                alert_state['detection_count'] = 0
                                
                                timer = threading.Timer(10.0, lambda cid=camera_id: circular_wave_alert_states[cid].update({
                                    'triggered': False,
                                    'detection_count': 0
                                }))
                                timer.daemon = True
                                timer.start()
                    else:
                        if camera_id in circular_wave_alert_states:
                            circular_wave_alert_states[camera_id]['detection_count'] = 0
                    
                    if action == 'circular_wave' and confidence > 0.7:
                        cv2.putText(display_frames[camera_id], f"CIRCULAR WAVE: {confidence:.2f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                except Exception as e:
                    print(f"Action recognition error: {e}")
        except concurrent.futures.TimeoutError:
            remaining = sum(1 for f in action_futures if not f.done())
            print(f"Frame update error: {remaining} (of {len(action_futures)}) futures unfinished")
    
    return display_frames

def update_camera_status():
    active_cameras = [cam_id for cam_id, cam in camera_manager.cameras.items() if cam.is_active]
    total_cameras = len(camera_manager.cameras)
    
    status_parts = []
    for camera_id, camera in camera_manager.cameras.items():
        if camera.is_active:
            status_parts.append(f"{camera_id}:✓")
        else:
            status_parts.append(f"{camera_id}:✗")
    
    status_text = f"Cameras: {len(active_cameras)}/{total_cameras} active | {' | '.join(status_parts)}"
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
#full screenshot function to take screenshots of all active cameras
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
    camera_manager.stop_all_cameras()
    executor.shutdown(wait=True)
    cv2.destroyAllWindows()
    root.destroy()

root = tk.Tk()
root.title("EyeSpy+")
root.geometry("1100x750")
root.configure(bg=DarkTheme.BG_PRIMARY)

DarkTheme.configure_styles(root)

main_frame = tk.Frame(root, bg=DarkTheme.BG_PRIMARY)
main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

title_label = tk.Label(main_frame, text="EyeSpy+",
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
                     text="Priority Cameras: Action Recognition + Gun Detection | Object Cameras: Gun Detection Only",
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

def start_cameras():
    Camera_Source_Selection(root, camera_manager)

start_btn = tk.Button(controls_frame, text="Start Cameras", 
                     command=start_cameras, **button_style)
start_btn.pack(side=tk.LEFT, padx=(0, 10))

stop_btn = tk.Button(controls_frame, text="Stop Cameras", 
                    command=camera_manager.stop_all_cameras, **button_style)
stop_btn.pack(side=tk.LEFT, padx=(0, 10))

screenshot_btn = tk.Button(controls_frame, text="Screenshot All", 
                          command=take_screenshot_all, **button_style)
screenshot_btn.pack(side=tk.LEFT, padx=(0, 10))

exit_btn = tk.Button(controls_frame, text="Exit", command=exit_app,
                    font=('Arial', 11, 'bold'), bg=DarkTheme.ERROR,
                    fg=DarkTheme.FG_PRIMARY, activebackground='#c0392b',
                    activeforeground=DarkTheme.FG_PRIMARY, bd=0, 
                    padx=20, pady=8, cursor='hand2')
exit_btn.pack(side=tk.RIGHT)

status_frame = tk.Frame(root, bg=DarkTheme.BG_SECONDARY)
status_frame.pack(fill=tk.X, side=tk.BOTTOM)

status_label = tk.Label(status_frame, text="Ready - No cameras active", 
                       bg=DarkTheme.BG_SECONDARY, fg=DarkTheme.FG_SECONDARY,
                       anchor='w', padx=10, pady=5)
status_label.pack(fill=tk.X)

from datetime import datetime

root.after(1000, update_frame)

root.protocol("WM_DELETE_WINDOW", exit_app)

root.mainloop()  
