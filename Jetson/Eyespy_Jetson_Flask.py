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
    
