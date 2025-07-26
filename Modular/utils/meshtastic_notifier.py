#EyeSpy+ Meshatastic Notifier Module
# Configured using Spec5 Ranger as MSG Rx and MiniTrekker as Tx

import meshtastic
import meshtastic.serial_interface


class MeshtasticNotifier:
    
    def __init__(self, device_path='COM3'):
        self.device_path = device_path
        self.interface = None
        self.connected = False
        self.connect()
    
    def connect(self):
        try:
            print(f"Connecting to Meshtastic on {self.device_path}...")
            self.interface = meshtastic.serial_interface.SerialInterface(self.device_path)
            self.connected = True
            print(f"Meshtastic connected on {self.device_path}")
            return True
        except Exception as e:
            print(f"Meshtastic connection failed: {e}")
            self.connected = False
            return False
    
    def send_alert(self, message):
        if not self.connected or not self.interface:
            return False
        #set limit for message length
        if len(message) > 230:
            message = message[:227] + "..."
        
        try:
            self.interface.sendText(message, channelIndex=1)
            print(f"Meshtastic sent: {message}")
            return True
        except Exception as e:
            print(f"Meshtastic send error: {e}")
            self.connected = False
            return False
    
    def close(self):
        if self.interface:
            try:
                self.interface.close()
            except:
                pass
        self.connected = False

_notifier = None

def setup_meshtastic(device_path='COM3'):
    global _notifier
    _notifier = MeshtasticNotifier(device_path)
    return _notifier.connected

def send_meshtastic_alert(message):
    if _notifier:
        return _notifier.send_alert(message)
    return False
