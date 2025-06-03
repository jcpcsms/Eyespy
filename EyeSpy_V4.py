#############################===== EYESPY+ V4.0, PAZ, J.C. FULL SAIL UNIVERSITY CSMS =====####################################################
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
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

####################################################################################################################################
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
#ActionRec neural net. 
class SkeletonActionClassifier(nn.Module):
    def __init__(self, input_size=66, hidden_size=256, num_layers=3, num_classes=5, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )
        
        self.lstm = nn.LSTM(
            hidden_size // 2, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        
        self.attention_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, 
            num_heads=self.attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        self.init_weights()
        
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape #tensor 
        
        x_proj = self.input_projection(x)
        lstm_out, _ = self.lstm(x_proj)  
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        lstm_pooled = torch.mean(lstm_out, dim=1)
        attention_pooled = torch.mean(attended_out, dim=1)
        combined_features = lstm_pooled + attention_pooled
        
        features = self.feature_extractor(combined_features)
        output = self.classifier(features)
        return output

####################################################################################################################################
#ActRec detections 
class ActionRecognition:
    def __init__(self):
        #ensure cuda is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        #Rev up MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, 
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.model_path = "models/mediapipe_action_classifier.pth"
        self.sequence_length = 60
        self.skeleton_sequence = deque(maxlen=self.sequence_length)
        self.action_classes = ['circular_wave', 'horizontal_wave', 'vertical_wave', 'stop_signal', 'idle']
        self.confidence_threshold = 0.7
        self.frame_skip_counter = 0
        self.process_every_n_frames = 1
        self.neural_classifier = None
        self.model_trained = False
        
        #load ActRec model
        print("Loading ActRec model...")
        self.load_model()
    
    def load_model(self):

        if not os.path.exists(self.model_path):
            print("Model not found >> ActRec disabled") #no bueno papasito
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.neural_classifier = SkeletonActionClassifier(
                input_size=66,
                num_layers=5,  
                num_classes=5 
            )
            
            self.neural_classifier.load_state_dict(checkpoint['model_state_dict'])
            self.neural_classifier.to(self.device)
            self.neural_classifier.eval()
            self.action_classes = ['circular_wave', 'horizontal_wave', 'vertical_wave', 'stop_signal', 'idle']
            self.model_trained = True
            print(f"ActRec model active on {self.device}!")
            return True
            
        except Exception as e:
            print("Action recognition disabled - only gun detection will work")
            self.model_trained = False
            return False
    
    def extract_pose_landmarks(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
                return landmarks
            
            return None
        except Exception as e:
            print(f"Error extracting skeleton pose landmarks: {e}")
            return None
    
    def normalize_pose_sequence(self, sequence):
        normalized = []
        
        for frame in sequence:
            if len(frame) < 66:
                frame = np.pad(frame, (0, 66 - len(frame)), 'constant')
            
            landmarks = np.array(frame[:66]).reshape(33, 2)
            
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
            
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
            hip_dist = np.linalg.norm(left_hip - right_hip)
            torso_height = np.linalg.norm((left_shoulder + right_shoulder)/2 - (left_hip + right_hip)/2)
            
            scale = max(shoulder_dist, hip_dist, torso_height) + 1e-6
            normalized_landmarks = (landmarks - center) / scale
            normalized.append(normalized_landmarks.flatten())
        
        return np.array(normalized)
    
    def predict_action(self):
        if not self.model_trained or len(self.skeleton_sequence) < 15:
            return 'idle', 0.5
        
        try:
            sequence = list(self.skeleton_sequence)
            normalized_sequence = self.normalize_pose_sequence(sequence)
            
            #sequence normalization for prediction
            if len(normalized_sequence) > self.sequence_length:
                start_idx = (len(normalized_sequence) - self.sequence_length) // 2
                processed_sequence = normalized_sequence[start_idx:start_idx + self.sequence_length]
            else:
                padding_needed = self.sequence_length - len(normalized_sequence)
                if len(normalized_sequence) > 0:
                    last_frame = normalized_sequence[-1]
                    processed_sequence = np.vstack([normalized_sequence] + [last_frame] * padding_needed)
                else:
                    return 'idle', 0.5
            
            #use CUDA if active
            sequence_tensor = torch.FloatTensor(processed_sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.neural_classifier(sequence_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_action = self.action_classes[predicted_idx.item()]
                return predicted_action, confidence.item()
                
        except Exception as e:
            print(f"Action prediction error: {e}")
            return 'idle', 0.5
    
    def process_frame(self, frame):
        landmarks = self.extract_pose_landmarks(frame)
        if landmarks:
            self.skeleton_sequence.append(landmarks)
            if len(self.skeleton_sequence) >= 15:
                predicted_action, confidence = self.predict_action()
                #show significant detections
                if confidence > self.confidence_threshold and predicted_action != 'idle':
                    print(f"Action detected: {predicted_action} (confidence: {confidence:.3f})")
                    return predicted_action, confidence    
        return None, 0.0

####################################################################################################################################
# Function 1. save_detection_log
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
# Function 2. cam_config
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
        print(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps} FPS")    
        
        return video_capture
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None

####################################################################################################################################
# Function 3. clean exit
def exit(video_capture):
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
    cv2.destroyAllWindows()
    root.destroy()

####################################################################################################################################
# Function 4. take_screenshot(including gun detection)
def take_screenshot():
    ret, frame = video_capture.read()
    if not ret:
        messagebox.showerror("Screenshot failure")
        return
    try:
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
# Function 5. nonblocking alert + play alarm sound
def show_nonblocking_alert(title, message):
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
        take_screenshot()
        alert_window.after(5000, alert_window.destroy)
        
    alert_thread = threading.Thread(target=show_alert)
    alert_thread.daemon = True 
    alert_thread.start()
      
#set global alert flags
gun_alert_triggered = False
circular_wave_alert_triggered = False
##########################################################################################################################################
# Function 6. process_gun_detections & triggering alert Gun detection from public dataset
# Gun detection frame counter for optimization
gun_detection_counter = 0

def process_gun_detections(frame):
    global gun_alert_triggered, gun_detection_counter
    
    #firearm detection every 3rd frame to save processing power for testing
    gun_detection_counter += 1
    if gun_detection_counter % 3 != 0:
        return frame
    
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
                        
                        if conf > 0.4:
                            print(f"Gun detection: {label} at coordinates ({x1}, {y1}, {x2}, {y2})")
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
                            save_detection_log(label, (x1, y1, x2-x1, y2-y1), "Weapon Model Detection")
                            
                            if gun_model.names[cls_id] == 'Gun' and conf > 0.65 and not gun_alert_triggered:
                                print("FIREARM ALERT TRIGGERED!")
                                show_nonblocking_alert("EyeSpy ALERT!", "FIREARM DETECTED!")
                                gun_alert_triggered = True
                                def reset_gun_alert_flag():
                                    global gun_alert_triggered
                                    gun_alert_triggered = False
                                timer = threading.Timer(10.0, reset_gun_alert_flag)
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
# Function 7. process_action_recognition - with circular wave alert
def process_action_recognition(frame):
    global circular_wave_alert_triggered
    if not action_recognizer.model_trained:
        return frame
    
    try:
        action, confidence = action_recognizer.process_frame(frame)
        
        if action and confidence > 0.7:  
            #display actions on live frame for debug ops
            cv2.putText(frame, f"Action: {action} ({confidence:.2f})", 
                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 0), 2)
            
            save_detection_log(f"{action}: {confidence:.2f}", "N/A", "Action Recognition")
            
            #Alert for positive circular_wave detection
            if action == 'circular_wave' and confidence > 0.75 and not circular_wave_alert_triggered:
                print("SOS ALERT TRIGGERED!")
                show_nonblocking_alert("EyeSpy ALERT!", "SOS DETECTED!")
                circular_wave_alert_triggered = True
                def reset_circular_wave_alert_flag():
                    global circular_wave_alert_triggered
                    circular_wave_alert_triggered = False
                timer = threading.Timer(10.0, reset_circular_wave_alert_flag)
                timer.daemon = True
                timer.start()
        
        return frame
    except Exception as e:
        print(f"Error in action recognition: {str(e)}")
        return frame

##########################################################################################################################################
# Function 8. update_frame for firearm and ActRec
def update_frame():
    try:
        ret, frame = video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Check camera.")
            return
    #mirror video for ActRec
        frame = cv2.flip(frame, 1)

        frame = process_gun_detections(frame)

        frame = process_action_recognition(frame)
        
        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_img)
        imgtk = ImageTk.PhotoImage(image=img)
        display_label.imgtk = imgtk
        display_label.configure(image=imgtk)
        display_label.after(50, update_frame) 

    except Exception as e:
        messagebox.showerror("Frame Update Error", f"Error during frame update: {str(e)}")

################################################################################################################################
root = tk.Tk()
root.title("Eyespy+")

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
# Initialize Action Recognition
action_recognizer = ActionRecognition()

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

#gun_model was provided by Felix Sam, https://github.com/Tech-Watt/Yolo11-Gun-Detection-Model/blob/main/model.pt
# Ref: https://github.com/felixchenfy/Realtime-Action-Recognition
# Ref: https://github.com/jeffreyyihuang/two-stream-action-recognition
# Ref: https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master
# Ref: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb 
