#matrix layout configuration for camera display in GUI
#this module creates a matrix display for camera feeds, handling USB camera mirroring and layout styling
import cv2
import numpy as np
from .styles import DarkTheme

# check if USB camera to flip for mirror effect, needed to match action recognition model
def should_flip_camera(camera_manager, camera_id):
    if camera_id in camera_manager.cameras:
        camera = camera_manager.cameras[camera_id]
        return camera.source_type == 'usb' 
    return False

def create_display_matrix(processed_frames, camera_manager):
    priority_width, priority_height = 400, 300  
    object_width, object_height = 266, 200     
    placeholder_priority = np.full((priority_height, priority_width, 3), 30, dtype=np.uint8)
    placeholder_object = np.full((object_height, object_width, 3), 30, dtype=np.uint8)

    cv2.putText(placeholder_priority, "No Signal", (priority_width//2-60, priority_height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    cv2.putText(placeholder_object, "No Signal", (object_width//2-50, object_height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    display_frames = {}
    
    # Process priority cameras
    for camera_id in ['priority_1', 'priority_2']:
        if camera_id in processed_frames:
            frame = processed_frames[camera_id]
            if should_flip_camera(camera_manager, camera_id):
                frame = cv2.flip(frame, 1)
            display_frames[camera_id] = cv2.resize(frame, (priority_width, priority_height))

            # Border and label
            cv2.rectangle(display_frames[camera_id], (0, 0), (priority_width-1, priority_height-1), 
                         (50, 50, 50), 2)
            
            # Label background
            cv2.rectangle(display_frames[camera_id], (0, 0), (150, 30), (30, 30, 30), -1)
            cv2.putText(display_frames[camera_id], f"{camera_id.upper()}", 
                       (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display_frames[camera_id] = placeholder_priority.copy()
            cv2.rectangle(display_frames[camera_id], (0, 0), (priority_width-1, priority_height-1), 
                         (50, 50, 50), 2)
            cv2.rectangle(display_frames[camera_id], (0, 0), (150, 30), (30, 30, 30), -1)
            cv2.putText(display_frames[camera_id], f"{camera_id.upper()}", 
                       (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    # Process object cameras
    for camera_id in ['object_1', 'object_2', 'object_3']:
        if camera_id in processed_frames:
            frame = processed_frames[camera_id]
            if should_flip_camera(camera_manager, camera_id):
                frame = cv2.flip(frame, 1)
            display_frames[camera_id] = cv2.resize(frame, (object_width, object_height))
            
            # Add border and label
            cv2.rectangle(display_frames[camera_id], (0, 0), (object_width-1, object_height-1), 
                         (50, 50, 50), 2)
            cv2.rectangle(display_frames[camera_id], (0, 0), (120, 25), (30, 30, 30), -1)
            cv2.putText(display_frames[camera_id], f"{camera_id.upper()}", 
                       (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            display_frames[camera_id] = placeholder_object.copy()
            cv2.rectangle(display_frames[camera_id], (0, 0), (object_width-1, object_height-1), 
                         (50, 50, 50), 2)
            cv2.rectangle(display_frames[camera_id], (0, 0), (120, 25), (30, 30, 30), -1)
            cv2.putText(display_frames[camera_id], f"{camera_id.upper()}", 
                       (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
    
    # Create matrix layout
    try:
        spacing = 2
        spacer_h = np.full((priority_height, spacing, 3), 20, dtype=np.uint8)
        spacer_v = np.full((spacing, priority_width * 2 + spacing, 3), 20, dtype=np.uint8)
        top_row = cv2.hconcat([display_frames['priority_1'], spacer_h, display_frames['priority_2']])
        spacer_h_small = np.full((object_height, spacing, 3), 20, dtype=np.uint8)
        bottom_row = cv2.hconcat([
            display_frames['object_1'], spacer_h_small,
            display_frames['object_2'], spacer_h_small,
            display_frames['object_3']
        ])
        
        # Ensure same width
        top_width = top_row.shape[1]
        bottom_width = bottom_row.shape[1]
        
        if top_width != bottom_width:
            bottom_row = cv2.resize(bottom_row, (top_width, bottom_row.shape[0]))

        matrix_display = cv2.vconcat([top_row, spacer_v, bottom_row])
        
        return matrix_display
        
    except Exception as e:
        print(f"Error creating display matrix: {e}")
        error_frame = np.full((600, 800, 3), 30, dtype=np.uint8)
        cv2.putText(error_frame, "Display Matrix Error", (250, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return error_frame
