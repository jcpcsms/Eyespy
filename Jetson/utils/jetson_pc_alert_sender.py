# EyeSpy+ Jetson toPC Alert Sender Module

import requests
import threading
import time

class JetsonPCAlertSender:
    
    def __init__(self, camera_id="jetson_local"):
        self.camera_id = camera_id
        self.pc_ip = "192.168.100.1"
        self.pc_port = 5001
        self.pc_alert_url = f"http://{self.pc_ip}:{self.pc_port}/jetson_alert"

        self.alerts_sent = 0
        self.alerts_successful = 0
        self.alerts_failed = 0
        
        print(f"JetsonPCAlertSender initialized for {camera_id}")
        print(f"   PC Target: {self.pc_ip}:{self.pc_port}")
    
    def send_sos_alert(self, action, confidence, duration):
        try:
            current_time = time.time()
            
            
            threading.Thread(
                target=self._send_sos_alert_threaded,
                args=(action, confidence, duration, current_time),
                daemon=True
            ).start()
            
            self.alerts_sent += 1
            print(f"SOS alert queued for PC: {action} ({confidence:.2f}) duration {duration:.1f}s")
            return True
            
        except Exception as e:
            print(f"Failed to queue SOS alert: {e}")
            self.alerts_failed += 1
            return False
    
    def _send_sos_alert_threaded(self, action, confidence, duration, timestamp):
        print(f"DEBUG: Starting HTTP POST to {self.pc_alert_url}") 
        send_start = time.time()
        
        try:
            alert_data = {
                'camera_id': self.camera_id,
                'type': 'action_detection',
                'action': action,
                'confidence': confidence,
                'duration': duration,
                'timestamp': timestamp,
                'message': f"SUSTAINED SOS DETECTED: {action} for {duration:.1f}s",
                'source': 'jetson_sustained_action'
            }
            
            
            response = requests.post(
                self.pc_alert_url, 
                json=alert_data, 
                timeout=3.0,
                headers={'Content-Type': 'application/json'}
            )
            
            request_time = (time.time() - send_start) * 1000
            
            if response.status_code == 200:
                print(f"SOS alert delivered to PC ({request_time:.1f}ms)")
                self.alerts_successful += 1
            else:
                print(f"PC responded with HTTP {response.status_code}")
                self.alerts_failed += 1
                
        except requests.exceptions.Timeout:
            print(f"PC alert timeout (check PC responsiveness & Firewall)")
            self.alerts_failed += 1
        except requests.exceptions.ConnectionError:
            print(f"PC connection failed (check PC: {self.pc_ip})")
            self.alerts_failed += 1
        except Exception as e:
            print(f"DEBUG: Exception details: {type(e).__name__}: {str(e)}")
            print(f"SOS alert failed: {e}")
            self.alerts_failed += 1
    
    def test_pc_connection(self):
        test_start = time.time()
        
        try:
            test_data = {
                'camera_id': self.camera_id,
                'type': 'connection_test',
                'confidence': 0.90,
                'timestamp': time.time(),
                'message': f"CONNECTION TEST from {self.camera_id}",
                'source': 'test'
            }
            
            response = requests.post(
                self.pc_alert_url, 
                json=test_data, 
                timeout=2.0
            )
            
            test_time = (time.time() - test_start) * 1000
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'message': 'PC connection successful',
                    'response_time_ms': test_time,
                    'pc_response': response.status_code
                }
            else:
                return {
                    'success': False,
                    'error': f'PC responded with HTTP {response.status_code}',
                    'response_time_ms': test_time
                }
                
        except requests.exceptions.Timeout:
            test_time = (time.time() - test_start) * 1000
            return {
                'success': False,
                'error': 'PC connection timeout',
                'response_time_ms': test_time
            }
        except requests.exceptions.ConnectionError:
            test_time = (time.time() - test_start) * 1000
            return {
                'success': False,
                'error': f'PC connection refused (check PC: {self.pc_ip})',
                'response_time_ms': test_time
            }
        except Exception as e:
            test_time = (time.time() - test_start) * 1000
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': test_time
            }
    
    def get_alert_statistics(self):
        success_rate = 0
        if self.alerts_sent > 0:
            success_rate = (self.alerts_successful / self.alerts_sent) * 100
        
        return {
            'alerts_sent': self.alerts_sent,
            'alerts_successful': self.alerts_successful,
            'alerts_failed': self.alerts_failed,
            'success_rate_percent': round(success_rate, 1),
            'pc_connection': {
                'ip': self.pc_ip,
                'port': self.pc_port,
                'url': self.pc_alert_url
            }
        }

_pc_alert_sender = None

def get_pc_alert_sender(camera_id="jetson_local"):
    global _pc_alert_sender
    if _pc_alert_sender is None:
        _pc_alert_sender = JetsonPCAlertSender(camera_id)
    return _pc_alert_sender
