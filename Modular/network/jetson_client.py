# Jetson client 
import socket
import json
import time
import threading
import struct
from collections import deque
import numpy as np

class JetsonClient:
    def __init__(self, jetson_ip='192.168.100.2', jetson_port=5555, timeout=0.1):
        self.jetson_ip = jetson_ip
        self.jetson_port = jetson_port
        self.timeout = timeout
        self.is_available = False
        self.last_check_time = 0
        self.check_interval = 5.0  # Check Jetson availability every 5 seconds
        
        # Initial availability check
        self.check_jetson_availability()
        
    def send_message(self, sock, data):
        json_bytes = json.dumps(data).encode('utf-8')
        length_header = struct.pack('>I', len(json_bytes))
        message = length_header + json_bytes
        sock.sendall(message)
        
    def receive_response(self, sock):
        length_data = sock.recv(4)
        if len(length_data) < 4:
            raise ValueError("Incomplete response header")
            
        response_length = struct.unpack('>I', length_data)[0]
        
        response_data = b''
        while len(response_data) < response_length:
            chunk = sock.recv(min(response_length - len(response_data), 4096))
            if not chunk:
                break
            response_data += chunk
            
        return json.loads(response_data.decode('utf-8'))
        
    def check_jetson_availability(self):
        current_time = time.time()

        if current_time - self.last_check_time < self.check_interval:
            return self.is_available
            
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect((self.jetson_ip, self.jetson_port))
                
                ping_data = {'type': 'ping'}
                self.send_message(s, ping_data)

                response = self.receive_response(s)
                
                if response.get('message') == 'pong':
                    if not self.is_available:
                        print(f"Jetson detected at {self.jetson_ip}:{self.jetson_port}")
                    self.is_available = True
                else:
                    self.is_available = False
                    
        except (socket.timeout, ConnectionRefusedError, OSError, ValueError) as e:
            if self.is_available:
                print(f"Jetson disconnected ({type(e).__name__}), falling back to local processing")
            self.is_available = False
            
        self.last_check_time = current_time
        return self.is_available
    
    #send landmarks to Jetson for action prediction
    def predict_action(self, landmarks_sequence, quality_level, temporal_quality, camera_id="cam0"):
        if not self.check_jetson_availability():
            return None 
            
        try:
            if not landmarks_sequence or len(landmarks_sequence) == 0:
                print(f"No landmarks to send for {camera_id}")
                return None
                
            # Prepare data packet
            data = {
                'type': 'predict',
                'camera_id': camera_id,
                'landmarks': [frame.tolist() if isinstance(frame, np.ndarray) else frame 
                             for frame in landmarks_sequence],
                'quality_level': quality_level,
                'temporal_quality': temporal_quality,
                'timestamp': time.time()
            }
            
            # Send to Jetson
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.jetson_ip, self.jetson_port))
 
                self.send_message(s, data)
                
                response = self.receive_response(s)
                
                if response['status'] == 'success':
                    action = response['action']
                    confidence = response['confidence']
                    if action == 'circular_wave':
                        print(f"SOS Detected from Jetson on {camera_id}! Confidence: {confidence:.2f}")
                    return action, confidence
                else:
                    print(f"Jetson error: {response.get('error', 'Unknown error')}")
                    return None
                    
        except socket.timeout:
            print(f"Jetson timeout for {camera_id}")
            self.is_available = False
            return None
        except (ConnectionError, json.JSONDecodeError, ValueError) as e:
            print(f"Jetson communication error for {camera_id}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.is_available = False
            return None
        except Exception as e:
            print(f"Unexpected error in Jetson client: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None


if __name__ == "__main__":
    print("Testing Jetson Client...")
    
    client = JetsonClient()
    
    if client.is_available:
        print("\n Sending test prediction...")
        test_landmarks = [np.random.randn(66).tolist() for _ in range(60)]
        
        result = client.predict_action(
            landmarks_sequence=test_landmarks,
            quality_level=4,
            temporal_quality=4,
            camera_id="test_camera"
        )
        
        if result:
            action, confidence = result
            print(f"Test successful! Action: {action}, Confidence: {confidence:.2f}")
        else:
            print("Test failed - no result received")
    else:
        print("Jetson not available")#placeholder
