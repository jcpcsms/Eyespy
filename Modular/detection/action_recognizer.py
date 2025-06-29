# action recognition module for detecting actions using MediaPipe Pose and updated resolution aware neural network classifier
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import os
from collections import deque
from network.jetson_client import JetsonClient

class ResolutionAwareActionClassifier(nn.Module):
    def __init__(self, input_size=66, hidden_size=256, num_layers=3, num_classes=5, dropout=0.4):
        super(ResolutionAwareActionClassifier, self).__init__()
        self.quality_embedding = nn.Embedding(6, 32)
        self.temporal_embedding = nn.Embedding(6, 16)
        self.input_projection = nn.Linear(input_size + 32 + 16, hidden_size)
        
        # LSTM layers with attention
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention mechanism for temporal features
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Dedicated priority detection
        self.circular_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, quality_level, temporal_quality):
        batch_size, seq_len, feature_size = x.shape
        
        # Quality embedding
        quality_emb = self.quality_embedding(quality_level) 
        quality_emb = quality_emb.unsqueeze(1).expand(-1, seq_len, -1) 
        
        # Temporal quality embedding  
        temporal_emb = self.temporal_embedding(temporal_quality)
        temporal_emb = temporal_emb.unsqueeze(1).expand(-1, seq_len, -1) 
        x = torch.cat([x, quality_emb, temporal_emb], dim=-1) 
        x = self.input_projection(x) 
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        lstm_out_transposed = lstm_out.transpose(0, 1) 
        attended, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attended = attended.transpose(0, 1)
        pooled = torch.mean(attended, dim=1)
        action_logits = self.classifier(pooled)
        circular_confidence = self.circular_detector(pooled)
        
        return action_logits, circular_confidence
##########################################################   
class ActionRecognition:
    def __init__(self, use_jetson=True, jetson_ip='192.168.100.2'):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose_configs = {
            'very_low': self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.1,
                min_tracking_confidence=0.1
            ),
            'low': self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.2,
                min_tracking_confidence=0.2
            ),
            'medium': self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            ),
            'high': self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        } 
        
        self.use_jetson = use_jetson
        if self.use_jetson:
            try:
                self.jetson_client = JetsonClient(jetson_ip=jetson_ip)
                if self.jetson_client.is_available:
                    print("Jetson AI acceleration enabled!")
                else:
                    print("Jetson not detected, using local processing")
            except Exception as e:
                print(f"Jetson client initialization failed: {e}")
                self.jetson_client = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.action_map = {'idle': 0,'circular_wave': 1,'horizontal_wave': 2,'vertical_wave': 3,'stop_signal': 4}
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        
        #Rev up MediaPipe
        self.mp_pose = mp.solutions.pose
        
        self.model_path = None
        self.sequence_length = 60
        self.skeleton_sequence = deque(maxlen=self.sequence_length)
        #self.action_classes = ['circular_wave', 'horizontal_wave', 'vertical_wave', 'stop_signal', 'idle']
        #self.confidence_threshold = 0.7
        #self.frame_skip_counter = 0
        #self.process_every_n_frames = 1
        self.neural_classifier = None
        self.min_fps_for_priority = 18
        self.model_trained = False
        
        #load ActRec model
        print("Loading ActRec model...")
        self.load_model()

    def set_camera_id(self, camera_id):
        self.camera_id = camera_id    

    def estimate_frame_quality(self, frame):
        height, width = frame.shape[:2]
        
        # Resolution-based quality
        pixel_count = width * height
        
        if pixel_count <= 320 * 240:
            return 0, 'very_low'
        elif pixel_count <= 640 * 480:
            return 2, 'low'
        elif pixel_count <= 960 * 720:
            return 3, 'medium'
        elif pixel_count <= 1280 * 720:
            return 4, 'high'
        else:
            return 5, 'high'   
    
    def load_model(self):
        
        self.model_path = "models/mediapipe_action_classifier_enhanced_temporal.pth"
        
        if not os.path.exists(self.model_path):
            print("Enhanced model not found >> ActRec disabled")
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            print("Loading enhanced temporal ActRec model...")
            self.neural_classifier = ResolutionAwareActionClassifier(
                input_size=66,
                hidden_size=256,
                num_layers=3, 
                num_classes=5,
                dropout=0.4
            )
            
            self.neural_classifier.load_state_dict(checkpoint['model_state_dict'])
            self.neural_classifier.to(self.device)
            self.neural_classifier.eval()
            print("Enhanced temporal ActRec model initialized.")

            if 'target_resolutions' in checkpoint:
                self.target_resolutions = checkpoint['target_resolutions']
            if 'target_frame_rates' in checkpoint:
                self.target_frame_rates = checkpoint['target_frame_rates']
            
            self.model_trained = True
            print(f"Enhanced temporal ActRec model active on {self.device}!")
            return True
            
        except Exception as e:
            print(f"Failed to load enhanced model: {e}")
            self.model_trained = False
            return False
    
    def extract_pose_landmarks(self, frame, config_key='high'):
        try:
            pose_processor = self.pose_configs[config_key]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_processor.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
                return landmarks
            
            return None
        except Exception as e:
            print(f"Error extracting skeleton pose landmarks: {e}")
            return None
    
    """def normalize_pose_sequence(self, sequence):
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
        
        return np.array(normalized)"""
    
    def predict_action(self, quality_level, temporal_quality=4):
        if self.use_jetson and self.jetson_client and len(self.skeleton_sequence) >= 30:
            jetson_result = self.jetson_client.predict_action(
                list(self.skeleton_sequence),
                quality_level,
                temporal_quality,
                self.camera_id
            )
            
            if jetson_result is not None:
                return jetson_result
        
        if self.neural_classifier is None or len(self.skeleton_sequence) < 30:
            return 'idle', 0.5
        
        # need enough frames for prediction
        sequence_length = len(self.skeleton_sequence)
        effective_fps = min(30, sequence_length / 2.0)

        # Reject SOS detection if frame rate too low
        if effective_fps < self.min_fps_for_priority:
            temporal_quality = max(0, temporal_quality - 2)
        
        # Prepare sequence for prediction
        sequence = np.array(list(self.skeleton_sequence)[-60:]) 
        if len(sequence) < 60:
            padding = np.tile(sequence[-1], (60 - len(sequence), 1))
            sequence = np.vstack([sequence, padding])

        # Convert to torch tensors
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        quality_tensor = torch.LongTensor([quality_level]).to(self.device)
        temporal_tensor = torch.LongTensor([temporal_quality]).to(self.device)
        
        # Model prediction
        with torch.no_grad():
            action_logits, circular_confidence = self.neural_classifier(sequence_tensor, quality_tensor, temporal_tensor)
            action_probs = F.softmax(action_logits, dim=1)
            
            # Enhanced priority logic with false positive protection
            circular_conf_value = circular_confidence.item()
            #print(f"DEBUG: Action probs: {action_probs.cpu().numpy()}")
            #print(f"DEBUG: Circular confidence: {circular_conf_value:.3f}")
            #print(f"DEBUG: Temporal quality: {temporal_quality}, FPS: {effective_fps}")
            # Adjust circular_wave threshold based on temporal quality
            if temporal_quality >= 3:  # Good temporal quality
                circular_threshold = 0.55  # Standard threshold
            elif temporal_quality == 2:  # Medium temporal quality  
                circular_threshold = 0.65  # Higher threshold
            else:  # Low temporal quality
                circular_threshold = 0.75  # Much higher threshold (avoid false positives)
            
            # Additional safety: require minimum confidence AND temporal quality
            if (circular_conf_value > circular_threshold and 
                temporal_quality >= 2 and 
                effective_fps >= 18):
                return 'circular_wave', circular_conf_value
            
            # General action prediction (with temporal quality consideration)
            predicted_class = torch.argmax(action_probs, dim=1).item()
            confidence = action_probs[0, predicted_class].item()
            
            # Reduce confidence for low temporal quality
            if temporal_quality <= 1:
                confidence *= 0.8 
            
            return self.reverse_action_map[predicted_class], confidence
    
    def process_frame(self, frame):
        quality_level, config_key = self.estimate_frame_quality(frame)
        landmarks = self.extract_pose_landmarks(frame, config_key)
        if landmarks:
            self.skeleton_sequence.append(landmarks)
            if len(self.skeleton_sequence) >= 30:
                predicted_action, confidence = self.predict_action(quality_level)

                if predicted_action == 'circular_wave' and confidence > 0.5:
                    print(f"PRIORITY ACTION: {predicted_action} (conf: {confidence:.3f})")    
                return predicted_action, confidence
        
        return 'idle', 0.0
