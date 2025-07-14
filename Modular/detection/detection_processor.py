#round robin detection processor for gun detection

import concurrent.futures
from utils.profiling_manager import is_profiling_active, log_profiling_event
import traceback

class DetectionProcessor:
    
    def __init__(self, gun_detector, executor=None):
        self.gun_detector = gun_detector
        self.executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=10)
        
        # Round-robin processing state
        self.gun_detection_frame_counter = 0
        self.current_camera_index = 0
        self.gun_detection_frame_interval = 1
        
        print("DetectionProcessor initialized")
    
    def get_round_robin_camera(self, camera_ids):
        self.gun_detection_frame_counter += 1

        if self.gun_detection_frame_counter % self.gun_detection_frame_interval != 0:
            return None

        camera_list = list(camera_ids)
        if len(camera_list) == 0:
            return None

        selected_camera = camera_list[self.current_camera_index % len(camera_list)]
        self.current_camera_index = (self.current_camera_index + 1) % len(camera_list)
        
        return selected_camera
    
    def process_all_cameras(self, all_frames):
        target_camera = self.get_round_robin_camera(all_frames.keys())
        
        if target_camera is None:
            return all_frames
        
        detection_futures = []
        
        # Process ONLY the target camera
        if target_camera in all_frames:
            frame = all_frames[target_camera]

            future = self.executor.submit(self.gun_detector.process_gun_detections, frame, target_camera)
            detection_futures.append((target_camera, future))

        for camera_id, future in detection_futures:
            try:
                annotated_frame, alert_needed, actual_confidence = future.result(timeout=2.0)
                all_frames[camera_id] = annotated_frame
                
                if alert_needed and is_profiling_active():
                    log_profiling_event('gun_detected', camera_id, actual_confidence)
                    
            except concurrent.futures.TimeoutError:
                print(f"Gun detection timeout for {camera_id}")
            except Exception as e:
                print(f"Gun detection error for {camera_id}: {e}")
                traceback.print_exc()

        
        return all_frames
    
    def set_detection_interval(self, interval):
        #Set gun detection frame interval (1 = every frame, 5 = every 5th frame)
        self.gun_detection_frame_interval = max(1, interval)
        print(f"Gun detection interval set to: {self.gun_detection_frame_interval}")
    
    def get_stats(self):
        """Get detection processor statistics"""
        return {
            'frame_counter': self.gun_detection_frame_counter,
            'current_camera_index': self.current_camera_index,
            'detection_interval': self.gun_detection_frame_interval
        }

_detection_processor_instance = None

def get_detection_processor(gun_detector=None, executor=None):
    global _detection_processor_instance
    if _detection_processor_instance is None:
        if gun_detector is None:
            raise ValueError("gun_detector must be provided for first init")
        _detection_processor_instance = DetectionProcessor(gun_detector, executor)
    return _detection_processor_instance
