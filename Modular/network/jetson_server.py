#================================ Eyespy+ Jestson Action Recognition Server====================================================#
# This server handles landmark data from clients, processes it using a resolution-aware action recognition model
import socket
import json
import threading
import numpy as np
import torch
import sys
import os
import struct
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.action_recognizer import ResolutionAwareActionClassifier

class JetsonActionServer:
    def __init__(self, host='0.0.0.0', port=5555):
        self.host = host
        self.port = port
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Jetson using device: {self.device}") # Needs to be CUDA!
        
        self.model = self.load_model()
        self.action_map = {
            'idle': 0, 'circular_wave': 1, 'horizontal_wave': 2,
            'vertical_wave': 3, 'stop_signal': 4
        }
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        
    def load_model(self):
        model_path = "models/mediapipe_action_classifier_enhanced_temporal.pth"
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            model = ResolutionAwareActionClassifier(
                input_size=66,
                hidden_size=256,
                num_layers=3,
                num_classes=5,
                dropout=0.4
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"Model loaded successfully on Jetson")
            return model
            
        except Exception as e:
            print(f"Failed to load model on Jetson: {e}")
            
            return None
    
    def process_request(self, data):
        try:
            request_type = data.get('type', 'predict')
            
            if request_type == 'ping':
                return {'status': 'success', 'message': 'pong'}
            
            if request_type == 'predict':
                if self.model is None:
                    return {'status': 'error', 'error': 'Model not loaded'}
                
                landmarks = data['landmarks']
                quality_level = data['quality_level']
                temporal_quality = data['temporal_quality']
                
                # Convert to numpy array and prepare sequence using last 60 frames
                sequence = np.array(landmarks[-60:]) 
                
                if len(sequence) < 60:
                    padding = np.tile(sequence[-1], (60 - len(sequence), 1))
                    sequence = np.vstack([sequence, padding])
                
                # Convert to tensors
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                quality_tensor = torch.LongTensor([quality_level]).to(self.device)
                temporal_tensor = torch.LongTensor([temporal_quality]).to(self.device)

                # Prediction
                with torch.no_grad():
                    action_logits, circular_confidence = self.model(
                        sequence_tensor, quality_tensor, temporal_tensor
                    )
                    action_probs = torch.nn.functional.softmax(action_logits, dim=1)
                
                    circular_conf_value = circular_confidence.item()
                    
                    # Action thresholds
                    if temporal_quality >= 3:
                        circular_threshold = 0.55
                    elif temporal_quality == 2:
                        circular_threshold = 0.65
                    else:
                        circular_threshold = 0.75
                    
                    if (circular_conf_value > circular_threshold and 
                        temporal_quality >= 2):
                        return {
                            'status': 'success',
                            'action': 'circular_wave',
                            'confidence': circular_conf_value
                        }
                    
                    predicted_class = torch.argmax(action_probs, dim=1).item()
                    confidence = action_probs[0, predicted_class].item()
                    
                    return {
                        'status': 'success',
                        'action': self.reverse_action_map[predicted_class],
                        'confidence': confidence
                    }
                    
        except Exception as e:
            print(f"Error processing request: {e}")
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}
    
    def handle_client(self, conn, addr):
        try:
            # Log connection
            print(f"New connection from {addr}")
            initial_peek = conn.recv(100, socket.MSG_PEEK)
            print(f"First 50 bytes: {initial_peek[:50]}")

            length_data = conn.recv(4)
            if not length_data or len(length_data) < 4:
                print(f"No length header from {addr}, got: {length_data}")
                return
            
            try:
                message_length = struct.unpack('>I', length_data)[0]
                print(f"Expected message length: {message_length} bytes")
                
                # 10 MB limit for message size
                if message_length > 10_000_000: 
                    print(f"Message too large: {message_length} bytes")
                    return
                    
            except struct.error as e:
                print(f"Invalid length header: {length_data} - {e}")
                traceback.print_exc()
                
                try:
                    all_data = length_data + conn.recv(4096)
                    request = json.loads(all_data.decode())
                    print("Parsed as legacy format (no length header)")
                except:
                    print(f"Cannot parse data from {addr}")
                    return
            else:
                data = b''
                while len(data) < message_length:
                    chunk = conn.recv(min(message_length - len(data), 4096))
                    if not chunk:
                        print(f"Connection closed while receiving data from {addr}")
                        break
                    data += chunk
                
                print(f"Received {len(data)} bytes of JSON data")
                
                # Parse JSON
                try:
                    request = json.loads(data.decode())
                    print(f"Parsed request type: {request.get('type', 'unknown')}")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Raw data: {data[:200]}...")
                    return
            
            response = self.process_request(request)
            print(f"Processing result: {response.get('status')} - {response.get('action', 'N/A')}")
            
            # Send response with length header
            response_data = json.dumps(response).encode()
            response_with_header = struct.pack('>I', len(response_data)) + response_data
            
            conn.sendall(response_with_header)
            print(f"Sent response: {len(response_data)} bytes")
            
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            traceback.print_exc()

            try:
                error_response = {'status': 'error', 'error': str(e)}
                error_data = json.dumps(error_response).encode()
                conn.sendall(struct.pack('>I', len(error_data)) + error_data)
            except:
                pass
        finally:
            conn.close()
            print(f"Closed connection from {addr}\n")
    
    def start(self):
        print("Starting Eyespy+ Jetson Action Recognition Server...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(5)
            
            print(f"Listening on {self.host}:{self.port}")
            print(f"Device: {self.device}")
            print(f"Debug mode: ON")
            print("Waiting for connections...\n")
            
            while True:
                conn, addr = s.accept()
                # threading - for handling multiple clients
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()

if __name__ == "__main__":
    server = JetsonActionServer()
    server.start()
