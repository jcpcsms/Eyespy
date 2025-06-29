import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
import os
import sys
import traceback

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.detection_logger import save_detection_log

class GunDetector:
    def __init__(self):
        self.gun_model = YOLO('models/model.pt')

        if torch.cuda.is_available():
            self.gun_model.to('cuda')
            print(f"YOLO Gun Model moved to GPU")
        
        print(f"YOLO Gun Model Device: {self.gun_model.device}")
        
        if torch.cuda.is_available():
            print(f"CUDA Available: {torch.cuda.device_count()} GPU(s)")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available - using CPU")

        #pre-fuse model to prevent repeating fusing errors
        try:
            self.gun_model.fuse()
        except Exception as e:
            print(f"Warning: Could not fuse model: {e}")
            traceback.print_exc()    
        
    def process_gun_detections(self, frame):
        try:
            results = self.gun_model(frame, verbose=False) 
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                            x1, y1, x2, y2 = map(int, coords)
                            
                            conf = float(box.conf[0]) if isinstance(box.conf, (list, tuple, np.ndarray)) else float(box.conf)
                            cls_id = int(box.cls[0]) if isinstance(box.cls, (list, tuple, np.ndarray)) else int(box.cls)
                            label = f"{self.gun_model.names[cls_id]}: {conf:.2f}"
                            
                            if conf > 0.4:
                                print(f"Gun detection: {label} at coordinates ({x1}, {y1}, {x2}, {y2})")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 2)
                                save_detection_log(label, (x1, y1, x2-x1, y2-y1), "Weapon Model Detection")

                                if self.gun_model.names[cls_id] == 'Gun' and conf > 0.65:
                                    print("FIREARM ALERT TRIGGERED!")
                                    return frame, True
                
                        except Exception as e:
                            print(f"Error processing gun detection box: {str(e)}")
                            traceback.print_exc()
                            continue

            return frame, False

        except Exception as e:
            print(f"Error in gun model detection: {str(e)}")
            traceback.print_exc()
            return frame, False
