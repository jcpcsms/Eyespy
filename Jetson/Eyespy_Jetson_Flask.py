#=============================================================== EyeSpy+ Jetson Orin Nano Edition========================================================================
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import threading
import queue
import time
import json
import os
import glob
import requests
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from flask import Flask, Response, render_template, jsonify, request
import base64
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import gc

###############################################################################
# force Jetson settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
    torch.cuda.set_per_process_memory_fraction(0.9)
    print(f"CUDA Device: {torch.cuda.get_device_name()}")

###################################################################################
#new resolution aware model for action classification
class ResolutionAwareActionClassifier(nn.Module):
    def __init__(self, input_size=66, hidden_size=256, num_layers=3, num_classes=5, dropout=0.4):
        super(ResolutionAwareActionClassifier, self).__init__()
        
        self.quality_embedding = nn.Embedding(6, 32)
        self.temporal_embedding = nn.Embedding(6, 16)
        self.input_projection = nn.Linear(input_size + 32 + 16, hidden_size)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.circular_detector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, quality_level, temporal_quality):
        batch_size, seq_len, feature_size = x.shape
        
        quality_emb = self.quality_embedding(quality_level)
        quality_emb = quality_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        temporal_emb = self.temporal_embedding(temporal_quality)
        temporal_emb = temporal_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        x = torch.cat([x, quality_emb, temporal_emb], dim=-1)
        x = self.input_projection(x)
        
        lstm_out, _ = self.lstm(x)
        
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attended, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attended = attended.transpose(0, 1)
        
        pooled = torch.mean(attended, dim=1)
        action_logits = self.classifier(pooled)
        circular_confidence = self.circular_detector(pooled)
        
        return action_logits, circular_confidence




    
