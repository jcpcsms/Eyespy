import cv2
import numpy as np
import torch
from ultralytics import YOLO
from utils.detection_logger import save_detection_log

class GunDetector:
    def __init__(self):
        print("Loading gun detection model...")
        model_path = "models/model.pt"
        try:
            self.gun_model = YOLO(model_path)

            if torch.cuda.is_available():
                self.gun_model.to('cuda')
                print("Gun Model loaded on GPU")
            else:
                print("Gun Model loaded on CPU")
                
            # Fuse model for better performance
            try:
                self.gun_model.fuse()
                print("Gun Model fused for optimal performance")
            except Exception as e:
                print(f"Could not fuse model: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Failed to load gun model: {e}")
            traceback.print_exc()
            self.gun_model = None
    
    def process_gun_detections(self, frame, camera_id="unknown"):
        if self.gun_model is None:
            return frame, False
            
        try:
            # Run YOLO inference
            results = self.gun_model(frame, verbose=False)
            
            detection_occurred = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            # Extract coordinates with error handling
                            coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                            x1, y1, x2, y2 = map(int, coords)
                            
                            # Extract confidence with error handling
                            conf = float(box.conf[0]) if isinstance(box.conf, (list, tuple, np.ndarray)) else float(box.conf)
                            cls_id = int(box.cls[0]) if isinstance(box.cls, (list, tuple, np.ndarray)) else int(box.cls)
                            
                            # Check confidence threshold
                            if conf > 0.60:
                                
                                print(f"GUN DETECTED [{camera_id}]: {conf:.0%} confidence at ({x1},{y1})-({x2},{y2})")
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, f'GUN: {conf:.0%}', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                
                     
                                save_detection_log('gun_detected', conf, f"Gun detected in {camera_id}", camera_id)
                                
                                detection_occurred = True
                                
                        except Exception as e:
                            print(f"Gun detection box processing error [{camera_id}]: {e}")
                            continue
            
            return frame, detection_occurred
            
        except Exception as e:
            print(f"Gun detection error [{camera_id}]: {e}")
            return frame, False
    
    def process_gun_detections_with_profiling(self, frame, camera_id="unknown", profiler=None):
        frame, detection_occurred = self.process_gun_detections(frame, camera_id)
        if profiler and detection_occurred:
            try:
                profiler.log_alert(f'gun_detected_{camera_id}')
            except Exception as e:
                print(f"Profiling error for {camera_id}: {e}")
                import traceback
                traceback.print_exc()
        
        return frame, detection_occurred
    
    def get_model_info(self):
        if self.gun_model is None:
            return {"status": "not_loaded", "device": "none"}
        
        device = "GPU" if torch.cuda.is_available() and next(self.gun_model.parameters()).is_cuda else "CPU"
        return {
            "status": "loaded",
            "device": device,
            "model_path": "models/model.pt"
        }
    
    def is_ready(self):
        return self.gun_model is not None
