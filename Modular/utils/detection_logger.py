# Log detection events with timestamp, label, coordinates, and detection type
import os
from datetime import datetime

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
