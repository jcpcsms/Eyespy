################################################## Real Time Action Recognition, JCP FSU CSMS #######################################################################
# Ref: https://github.com/felixchenfy/Realtime-Action-Recognition
# Ref: https://github.com/jeffreyyihuang/two-stream-action-recognition
# Ref: https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master

import cv2
import numpy as np
import json
import time
import os
from datetime import datetime
from collections import deque
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

######################################################################################################################################################
class ActionDataset(Dataset):
    def __init__(self, sequences, labels, sequence_length=30, augment=True):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
        self.augment = augment
        self.processed_sequences = self.preprocess_sequences(sequences)
        
    def preprocess_sequences(self, sequences):
        processed = []
        for sequence in sequences:
            if len(sequence) < 5:
                continue
            seq_array = np.array(sequence)
            

            normalized_seq = self.normalize_pose_sequence(seq_array)
            
            
            smoothed_seq = self.smooth_sequence(normalized_seq)
            
            processed.append(smoothed_seq.tolist())
        
        return processed
    
    def normalize_pose_sequence(self, sequence):
        normalized = []
        
        for frame in sequence:
            if len(frame) < 66:  # 33 landmarks * 2 coords
                # Pad if needed
                frame = np.pad(frame, (0, 66 - len(frame)), 'constant')
            
            # Reshape to (33, 2) for x,y coordinates
            landmarks = np.array(frame[:66]).reshape(33, 2)
            
            # Use shoulders and hips as reference points for normalization
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calculate torso center and scale
            center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
            
            # Calculate torso size for scale normalization
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
            hip_dist = np.linalg.norm(left_hip - right_hip)
            torso_height = np.linalg.norm((left_shoulder + right_shoulder)/2 - (left_hip + right_hip)/2)
            
            scale = max(shoulder_dist, hip_dist, torso_height) + 1e-6
            
            # Normalize: center around torso and scale by torso size
            normalized_landmarks = (landmarks - center) / scale
            
            normalized.append(normalized_landmarks.flatten())
        
        return np.array(normalized)
    
    def smooth_sequence(self, sequence, window_size=3):
        if len(sequence) < window_size:
            return sequence
        
        smoothed = np.zeros_like(sequence)
        for i in range(len(sequence)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(sequence), i + window_size // 2 + 1)
            smoothed[i] = np.mean(sequence[start_idx:end_idx], axis=0)
        
        return smoothed
    
    def augment_sequence(self, sequence):
        if not self.augment:
            return sequence
        
        aug_sequence = np.array(sequence)

        if np.random.random() < 0.3:
            scale_factor = np.random.uniform(0.8, 1.2)
            new_length = int(len(aug_sequence) * scale_factor)
            if new_length > 5:
                indices = np.linspace(0, len(aug_sequence) - 1, new_length)
                aug_sequence = np.array([aug_sequence[int(i)] for i in indices])
 
        if np.random.random() < 0.4:
            angle = np.random.uniform(-0.2, 0.2)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            for i in range(len(aug_sequence)):
                landmarks = aug_sequence[i].reshape(33, 2)
                rotated_landmarks = landmarks @ rotation_matrix.T
                aug_sequence[i] = rotated_landmarks.flatten()

        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, aug_sequence.shape)
            aug_sequence += noise

        if np.random.random() < 0.3:
            landmarks = aug_sequence.reshape(len(aug_sequence), 33, 2)
            landmarks[:, :, 0] = -landmarks[:, :, 0]  
            aug_sequence = landmarks.reshape(len(aug_sequence), 66)
        
        return aug_sequence.tolist()
        
    def __len__(self):
        return len(self.processed_sequences)
    
    def __getitem__(self, idx):
        sequence = self.processed_sequences[idx]
        label = self.labels[idx]

        sequence = self.augment_sequence(sequence)

        if len(sequence) > self.sequence_length:
            if self.augment:
                max_start = len(sequence) - self.sequence_length
                start_idx = np.random.randint(0, max_start + 1)
            else:
                start_idx = (len(sequence) - self.sequence_length) // 2
            sequence = sequence[start_idx:start_idx + self.sequence_length]
        else:
            padding_needed = self.sequence_length - len(sequence)
            if len(sequence) > 0:
                last_frame = sequence[-1]
                sequence = sequence + [last_frame] * padding_needed
            else:
                sequence = [[0] * 66] * self.sequence_length
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])
#######################################################################################################################################################
class SkeletonActionClassifier(nn.Module):
    def __init__(self, input_size=66, hidden_size=256, num_layers=3, num_classes=5, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #input projection to reduce noise and extract features
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )
        
        #Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_size // 2, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=True
        )
        
        #Multi-head attention for feature extraction
        self.attention_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, 
            num_heads=self.attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        #feature extractor
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
        """Initialize weights with Xavier/Glorot initialization"""
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
        batch_size, seq_len, _ = x.shape
        
        x_proj = self.input_projection(x)
        
        lstm_out, _ = self.lstm(x_proj)  
        
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        lstm_pooled = torch.mean(lstm_out, dim=1)
        attention_pooled = torch.mean(attended_out, dim=1)
        combined_features = lstm_pooled + attention_pooled
        
       
        features = self.feature_extractor(combined_features)
        
        output = self.classifier(features)
        return output
#######################################################################################################################################################
class SimpleActionClassifier:
    """Simple rule-based classifier as fallback"""
    def __init__(self):
        self.action_classes = ['circular_wave', 'horizontal_wave', 'vertical_wave', 'stop_signal', 'idle']
        
    def predict(self, skeleton_sequence):
        """Simple rule-based prediction"""
        if len(skeleton_sequence) < 10:
            return 'idle', 0.5
        
        features = self.extract_simple_features(self.skeleton_sequence)
        
        if features['circular_score'] > 0.6:
            return 'circular_wave', features['circular_score']
        elif features['horizontal_motion'] > 0.7:
            return 'horizontal_wave', features['horizontal_motion']
        elif features['vertical_motion'] > 0.7:
            return 'vertical_wave', features['vertical_motion']
        elif features['movement_variance'] < 0.1:
            return 'stop_signal', 0.8
        else:
            return 'idle', 0.6
    
    def extract_simple_features(self, skeleton_sequence):
        """Extract simple motion features"""
        if len(skeleton_sequence) < 2:
            return {'circular_score': 0, 'horizontal_motion': 0, 'vertical_motion': 0, 'movement_variance': 0}
        
        wrist_positions = []
        for skeleton in skeleton_sequence:
            if len(skeleton) >= 32:
                wrist_positions.append([skeleton[30], skeleton[31]])
        
        if len(wrist_positions) < 5:
            return {'circular_score': 0, 'horizontal_motion': 0, 'vertical_motion': 0, 'movement_variance': 0}
        
        wrist_positions = np.array(wrist_positions)
        
        x_motion = np.std(wrist_positions[:, 0])
        y_motion = np.std(wrist_positions[:, 1])
        
        center = np.mean(wrist_positions, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in wrist_positions]
        radius_consistency = 1.0 - (np.std(distances) / (np.mean(distances) + 1e-6))
        
        angles = [np.arctan2(pos[1] - center[1], pos[0] - center[0]) for pos in wrist_positions]
        angle_coverage = len(set(np.round(np.array(angles) * 4 / np.pi))) / 8.0 
        
        circular_score = (radius_consistency + angle_coverage) / 2.0
        
        return {
            'circular_score': circular_score,
            'horizontal_motion': min(x_motion * 2, 1.0),
            'vertical_motion': min(y_motion * 2, 1.0),
            'movement_variance': min((x_motion + y_motion) * 2, 1.0)
        }
#######################################################################################################################################################
class MediaPipeActionRecognition:
    def __init__(self):
        
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Action recognition settings
        self.sequence_length = 60
        self.skeleton_sequence = deque(maxlen=self.sequence_length)
        self.action_history = deque(maxlen=10)
        
        self.key_landmarks = [
            #body landmarks for gesture recognition
            11, 12,  # Shoulders
            13, 14,  # Elbows
            15, 16,  # Wrists
            17, 18,  # Pinkies
            19, 20,  # Index fingers
            21, 22,  # Thumbs
            23, 24,  # Hips
            0,       # Nose
        ]
        
        
        self.simple_classifier = SimpleActionClassifier()  
        self.neural_classifier = SkeletonActionClassifier(input_size=66)  
        self.use_neural_net = False 
        self.model_trained = False
        self.recording = False
        self.countdown_active = False
        self.countdown_start_time = 0
        self.countdown_duration = 3.0
        self.recording_start_time = 0
        self.current_action_data = []
        self.action_name = ""
        self.max_recording_time = 10
        
        self.system_state = "TRAINING"  # TRAINING, READY, RECOGNIZING
        
        self.data_folder = "action_data"
        self.model_folder = "models"
        self.ensure_folders()
        
        #action classes
        self.action_classes = ['circular_wave', 'horizontal_wave', 'vertical_wave', 'stop_signal', 'idle']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_classes)}
        
        self.confidence_threshold = 0.7
        self.action_cooldown = 2.0
        self.last_action_time = 0
        
        self.load_model()
        
    def ensure_folders(self):
        for folder in [self.data_folder, self.model_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def extract_pose_landmarks(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract all 33 landmarks
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
            
            return landmarks, results
        
        return None, results
    
    def draw_pose_landmarks(self, frame, results):
        if results.pose_landmarks:
            #landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            #draw skeleton key points
            height, width = frame.shape[:2]
            for idx in self.key_landmarks:
                if idx < len(results.pose_landmarks.landmark):
                    landmark = results.pose_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), -1) 
    
    def load_training_data(self):
        """Load all training data from files"""
        if not os.path.exists(self.data_folder):
            return [], []
        
        sequences = []
        labels = []
        
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(self.data_folder, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        action_name = data['action_name']
                        skeleton_sequence = data['skeleton_sequence']
                        
                        if action_name in self.action_to_idx and len(skeleton_sequence) > 10:
                            sequences.append(skeleton_sequence)
                            labels.append(self.action_to_idx[action_name])
                            
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return sequences, labels
    
    def train_model(self):
        self.system_state = "READY"
        
        sequences, labels = self.load_training_data()
        
        if len(sequences) < 5:
            print("Not enough training data. Needs 5 examples minimum.")
            return False
        
        print(f"Training with {len(sequences)} examples")
        
        if sequences:
            input_size = len(sequences[0][0]) if sequences[0] else 66
            print(f"Input size: {input_size}")
        else:
            input_size = 66
        
        #boost training data
        train_dataset = ActionDataset(sequences, labels, self.sequence_length, augment=True)
        val_dataset = ActionDataset(sequences, labels, self.sequence_length, augment=False)
        
    
        if len(sequences) >= 10:
            from collections import Counter
            label_counts = Counter(labels)
            
            train_indices = []
            val_indices = []
            
            for label_idx in range(len(self.action_classes)):
                class_indices = [i for i, label in enumerate(labels) if label == label_idx]
                if len(class_indices) >= 3:
                    #splits
                    val_size = max(1, len(class_indices) // 5)
                    val_indices.extend(class_indices[:val_size])
                    train_indices.extend(class_indices[val_size:])
                else:
                    train_indices.extend(class_indices)
            
            if val_indices:
                train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
                val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
            else:
                val_dataset = None
        else:
            val_dataset = None
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(16, len(train_dataset)), 
            shuffle=True,
            num_workers=0, 
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=min(16, len(val_dataset)), 
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        
        num_classes = len(self.action_classes)
        self.neural_classifier = SkeletonActionClassifier(
            input_size=input_size, 
            hidden_size=256, 
            num_layers=5,
            num_classes=num_classes,
            dropout=0.4  
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
        
        #OPTIMIZE PRIME!!!!!!
        optimizer = optim.AdamW([
            {'params': self.neural_classifier.input_projection.parameters(), 'lr': 0.001},
            {'params': self.neural_classifier.lstm.parameters(), 'lr': 0.0005},
            {'params': self.neural_classifier.attention.parameters(), 'lr': 0.001},
            {'params': self.neural_classifier.classifier.parameters(), 'lr': 0.001}
        ], weight_decay=0.01)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        num_epochs = 150  
        best_val_acc = 0
        best_train_acc = 0
        patience_counter = 0
        early_stopping_patience = 30
        
        print("OPTIMIZED TRAINING WITH BOOST! LETS GO!")
        
        for epoch in range(num_epochs):
            self.neural_classifier.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for sequences_batch, labels_batch in train_loader:
                optimizer.zero_grad()
                
                outputs = self.neural_classifier(sequences_batch)
                loss = criterion(outputs, labels_batch.squeeze())
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.neural_classifier.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch.squeeze()).sum().item()
            
            train_acc = 100 * train_correct / train_total
            
            #validation
            val_acc = 0
            if val_loader and len(val_loader) > 0:
                self.neural_classifier.eval()
                val_correct = 0
                val_total = 0
                val_loss = 0
                
                with torch.no_grad():
                    for sequences_batch, labels_batch in val_loader:
                        outputs = self.neural_classifier(sequences_batch)
                        loss = criterion(outputs, labels_batch.squeeze())
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels_batch.size(0)
                        val_correct += (predicted == labels_batch.squeeze()).sum().item()
                
                val_acc = 100 * val_correct / val_total if val_total > 0 else 0
                
                scheduler.step(val_acc)
                
                #save top model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.save_model()
                else:
                    patience_counter += 1
            else:
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    patience_counter = 0
                    self.save_model()
                else:
                    patience_counter += 1
            
            if epoch % 10 == 0 or epoch < 20:
                val_text = f", Val Acc: {val_acc:.1f}%" if val_loader else ""
                print(f"Epoch {epoch:3d}: Train Acc: {train_acc:.1f}%{val_text}, Loss: {train_loss/len(train_loader):.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        #evaluate best model
        if val_loader:
            print(f"Training completed! Best validation accuracy: {best_val_acc:.1f}%")
        else:
            print(f"Training completed! Best training accuracy: {best_train_acc:.1f}%")
        
        self.neural_classifier.eval()
        self.evaluate_model(train_loader, val_loader)
        
        self.model_trained = True
        self.use_neural_net = True
        self.system_state = "READY"
        
        return True
    
    def evaluate_model(self, train_loader, val_loader=None):
        from collections import defaultdict
        
        def evaluate_loader(loader, name):
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for sequences_batch, labels_batch in loader:
                    outputs = self.neural_classifier(sequences_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    predictions.extend(predicted.cpu().numpy())
                    true_labels.extend(labels_batch.squeeze().cpu().numpy())
            
            class_correct = defaultdict(int)
            class_total = defaultdict(int)
            
            for pred, true in zip(predictions, true_labels):
                class_total[true] += 1
                if pred == true:
                    class_correct[true] += 1
            
            print(f"\n{name} Results:")
            for class_idx, class_name in enumerate(self.action_classes):
                if class_total[class_idx] > 0:
                    acc = 100 * class_correct[class_idx] / class_total[class_idx]
                    print(f"  {class_name}: {acc:.1f}% ({class_correct[class_idx]}/{class_total[class_idx]})")
        
        evaluate_loader(train_loader, "Training")
        if val_loader:
            evaluate_loader(val_loader, "Validation")
    
    def save_model(self):
        """Save the trained model"""
        if not self.neural_classifier:
            return
        
        model_path = os.path.join(self.model_folder, "mediapipe_action_classifier.pth")
        torch.save({
            'model_state_dict': self.neural_classifier.state_dict(),
            'action_classes': self.action_classes,
            'sequence_length': self.sequence_length,
            'input_size': self.neural_classifier.lstm.input_size
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        model_path = os.path.join(self.model_folder, "mediapipe_action_classifier.pth") #delete this model if score is weak
        
        if not os.path.exists(model_path):
            return False
        
        try:
            checkpoint = torch.load(model_path)
            num_classes = len(checkpoint['action_classes'])
            input_size = checkpoint.get('input_size', 66)
            
            self.neural_classifier = SkeletonActionClassifier(
                input_size=input_size,
                num_classes=num_classes
            )
            self.neural_classifier.load_state_dict(checkpoint['model_state_dict'])
            self.neural_classifier.eval()
            
            self.action_classes = checkpoint['action_classes']
            self.action_to_idx = {action: idx for idx, action in enumerate(self.action_classes)}
            self.sequence_length = checkpoint.get('sequence_length', 30)
            
            self.model_trained = True
            self.use_neural_net = True
            self.system_state = "READY"
            
            print("Pre-trained model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict_action(self):
        if len(self.skeleton_sequence) < 10:
            return 'idle', 0.5
        
        # Convert to format expected by classifier
        sequence = list(self.skeleton_sequence)
        
        if self.use_neural_net and self.neural_classifier and self.model_trained:
            try:
                processed_sequence = self.preprocess_sequence_for_prediction(sequence)
                
                if len(processed_sequence) > self.sequence_length:
                    start_idx = (len(processed_sequence) - self.sequence_length) // 2
                    processed_sequence = processed_sequence[start_idx:start_idx + self.sequence_length]
                else:
                    padding_needed = self.sequence_length - len(processed_sequence)
                    if processed_sequence:
                        last_frame = processed_sequence[-1]
                        processed_sequence = processed_sequence + [last_frame] * padding_needed
                    else:
                        return 'idle', 0.5
                
                sequence_tensor = torch.FloatTensor(processed_sequence).unsqueeze(0)
                
                with torch.no_grad():
                    self.neural_classifier.eval()
                    output = self.neural_classifier(sequence_tensor)
                    probabilities = F.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    #debug info
                    probs = probabilities[0].cpu().numpy()
                    if hasattr(self, 'debug_counter'):
                        self.debug_counter += 1
                    else:
                        self.debug_counter = 0
                    
                    #print debug info
                    if self.debug_counter % 30 == 0:
                        print(f"Debug - Probabilities: {[f'{self.action_classes[i]}:{probs[i]:.3f}' for i in range(len(probs))]}")
                    
                    predicted_action = self.action_classes[predicted_idx.item()]
                    return predicted_action, confidence.item()
                    
            except Exception as e:
                print(f"Neural network prediction failed: {e}")
                # Fall back to simple classifier
                self.use_neural_net = False
        
        # Simple rule-based prediction
        return self.simple_classifier.predict(sequence)
    
    def preprocess_sequence_for_prediction(self, sequence):
        if len(sequence) < 5:
            return sequence
        
        # Convert to numpy for easier processing
        seq_array = np.array(sequence)
        
        # Normalize based on torso (same as training)
        normalized_seq = self.normalize_pose_sequence_prediction(seq_array)
        
        # Smooth the sequence to reduce noise (same as training)
        smoothed_seq = self.smooth_sequence_prediction(normalized_seq)
        
        return smoothed_seq.tolist()
    
    def normalize_pose_sequence_prediction(self, sequence):
        normalized = []
        
        for frame in sequence:
            if len(frame) < 66:  # 33 landmarks * 2 coords
                # Pad if needed
                frame = np.pad(frame, (0, 66 - len(frame)), 'constant')
            
            # Reshape to (33, 2) for x,y coordinates
            landmarks = np.array(frame[:66]).reshape(33, 2)
            
            #Use shoulders and hips as reference points for normalization
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calculate torso center and scale
            center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4
            
            #Calculate torso size for scale normalization
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
            hip_dist = np.linalg.norm(left_hip - right_hip)
            torso_height = np.linalg.norm((left_shoulder + right_shoulder)/2 - (left_hip + right_hip)/2)
            
            scale = max(shoulder_dist, hip_dist, torso_height) + 1e-6
            
            normalized_landmarks = (landmarks - center) / scale
            
            normalized.append(normalized_landmarks.flatten())
        
        return np.array(normalized)
    
    def smooth_sequence_prediction(self, sequence, window_size=3):
        if len(sequence) < window_size:
            return sequence
        
        smoothed = np.zeros_like(sequence)
        for i in range(len(sequence)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(sequence), i + window_size // 2 + 1)
            smoothed[i] = np.mean(sequence[start_idx:end_idx], axis=0)
        
        return smoothed
    
    def start_recording(self, action_name):
        """Start recording action"""
        self.action_name = action_name
        self.countdown_start_time = time.time()
        self.countdown_active = True
        self.recording = False
        self.current_action_data = []
        print(f"Get ready to perform '{action_name}'!")
    
    def stop_recording(self):
        if not self.recording and not self.countdown_active:
            return False
        
        if self.countdown_active:
            self.countdown_active = False
            print("Recording cancelled")
            return False
        
        self.recording = False
        
        if len(self.current_action_data) < 20:
            print("Recording too short")
            return False
        
        # Save recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_folder}/{self.action_name}_{timestamp}.json"
        
        recording_data = {
            'action_name': self.action_name,
            'timestamp': timestamp,
            'skeleton_sequence': self.current_action_data,
            'duration': time.time() - self.recording_start_time
        }
        
        with open(filename, 'w') as f:
            json.dump(recording_data, f)
        
        print(f"Saved: {filename}")
        print(f"Pose frames: {len(self.current_action_data)}")
        return True
    
    def get_training_data_count(self):
        if not os.path.exists(self.data_folder):
            return {}
        
        training_files = [f for f in os.listdir(self.data_folder) if f.endswith('.json')]
        counts = {}
        
        for filename in training_files:
            parts = filename.split('_')
            if len(parts) >= 2:
                action_name = parts[0]
                counts[action_name] = counts.get(action_name, 0) + 1
        
        return counts
    
    def check_ready_for_training(self):
        counts = self.get_training_data_count()
        if len(counts) >= 3 and all(count >= 2 for count in counts.values()):
            return True
        return False
    
    def process_frame(self, frame):
        current_time = time.time()
        
        #countdown
        if self.countdown_active:
            elapsed = current_time - self.countdown_start_time
            remaining = self.countdown_duration - elapsed
            
            if remaining > 0:
                countdown_num = int(remaining) + 1
                if countdown_num <= 3:
                    cv2.putText(frame, str(countdown_num), 
                               (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
                    cv2.putText(frame, f"Get ready for {self.action_name}!", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return frame
            else:
                self.countdown_active = False
                self.recording = True
                self.recording_start_time = current_time
                print(f" Recording '{self.action_name}' - GO!")
        
        landmarks, pose_results = self.extract_pose_landmarks(frame)
        
        if landmarks:
            self.skeleton_sequence.append(landmarks)
            
            if self.recording:
                self.current_action_data.append(landmarks)
            
            self.draw_pose_landmarks(frame, pose_results)
            
            #Only predict if system is ready and not recording
            if (self.system_state == "READY" and not self.recording and not self.countdown_active and 
                len(self.skeleton_sequence) >= 15):
                
                predicted_action, confidence = self.predict_action()
                
                #Add to action history for smoothing
                self.action_history.append((predicted_action, confidence))
                
                #Get most common recent action
                if len(self.action_history) >= 5:
                    recent_actions = [action for action, conf in list(self.action_history)[-5:] 
                                    if conf > self.confidence_threshold]
                    if recent_actions:
                        most_common = max(set(recent_actions), key=recent_actions.count)
                        if (most_common != 'idle' and 
                            current_time - self.last_action_time > self.action_cooldown):
                            
                            print(f"Action recognized: {most_common} (confidence: {confidence:.2f})")
                            
                            self.last_action_time = current_time
                
                #Predictions
                cv2.putText(frame, f"Action: {predicted_action} ({confidence:.2f})", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        else:
            #No pose detected
            cv2.putText(frame, "No Action Detected", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        #Handle recording display
        if self.recording:
            elapsed = current_time - self.recording_start_time
            if elapsed > self.max_recording_time:
                self.stop_recording()
            
            cv2.putText(frame, f"RECORDING: {self.action_name}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Frames: {len(self.current_action_data)}", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
#######################################################################################################################################################
def main():
    print("ACTION TRAINING AND RECOGNITION SYSTEM")
    
    recognizer = MediaPipeActionRecognition()
    
    #Setup camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #Actions to train
    actions = {
        ord('1'): "circular_wave",
        ord('2'): "horizontal_wave", 
        ord('3'): "vertical_wave",
        ord('4'): "stop_signal",
        ord('5'): "idle"
    }
    
    print("\nMEDIAPIPE ACTION RECOGNITION STARTED")
    print("Controls:")
    print("  1-5: Record action examples")
    print("  T: Train model (after recording examples)")
    print("  X: Stop recording")
    print("  Q: Quit")

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        

        processed_frame = recognizer.process_frame(frame)
        
        #Add system state
        if not recognizer.countdown_active and not recognizer.recording:
            training_counts = recognizer.get_training_data_count()
            
            #System state indicator
            state_color = (0, 255, 255) if recognizer.system_state == "TRAINING" else (0, 255, 0)
            cv2.putText(processed_frame, f"STATE: {recognizer.system_state}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            
            if recognizer.system_state == "TRAINING":
                cv2.putText(processed_frame, "Record examples (1-5), then press T to train", 
                           (10, processed_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                #Show training data counts
                count_text = f"Training data: {training_counts}"
                cv2.putText(processed_frame, count_text, 
                           (10, processed_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                #Check if ready for training
                if recognizer.check_ready_for_training():
                    cv2.putText(processed_frame, "Ready for training! Press 'T'", 
                               (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(processed_frame, "Need more examples for training", 
                               (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            elif recognizer.system_state == "READY":
                cv2.putText(processed_frame, "RECOGNITION MODE: Perform gestures for detection!", 
                           (10, processed_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(processed_frame, "Press 1-5 to record more examples, T to retrain", 
                           (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('MediaPipe Action Recognition', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in actions and not recognizer.recording and not recognizer.countdown_active:
            recognizer.start_recording(actions[key])
        elif key == ord('x'):
            recognizer.stop_recording()
        elif key == ord('t') and recognizer.system_state == "TRAINING":
            print("Starting training process...")
            if recognizer.check_ready_for_training():
                success = recognizer.train_model()
                if success:
                    print("Training completed! System ready for recognition.")
                else:
                    print("Training failed.")
            else:
                print("Not enough training data. Record more examples.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
