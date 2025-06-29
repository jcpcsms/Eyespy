#single camera source module
import queue

class CamSource:
    def __init__(self, camera_id, source_type, source_path, role='object', config=None):
        self.camera_id = camera_id
        self.source_type = source_type 
        self.source_path = source_path
        self.role = role
        self.config = config or {}
        self.is_active = False
        self.capture = None
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.last_frame = None
        # Set target frame rate for Object Cams to 15 FPS to preserve resources
        # Priority Cams will run at 30 FPS
        self.fps_target = self.config.get('fps', 30 if role == 'priority' else 15)
        self.loop_video = self.config.get('loop', True)
