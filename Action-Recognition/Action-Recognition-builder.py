import cv2 
import numpy as np 
import json
import time
import os
from datetime import datetime
from collections import deque
import mediapipe as mp 
import torch
import torch optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


PYTORCH_AVAILABLE = True
MEDIAPIPE_AVAILABLE = True
