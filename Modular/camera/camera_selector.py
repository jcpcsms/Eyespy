# Camera set up / source selection window for configuring camera sources
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from gui.styles import DarkTheme

class Camera_Source_Selection:
    def __init__(self, parent, camera_manager):
        self.parent = parent
        self.camera_manager = camera_manager
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Camera Source Selection")
        self.dialog.geometry("700x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.dialog.configure(bg=DarkTheme.BG_PRIMARY)
        
        self.camera_configs = {}
        self.create_interface()
        
    def create_interface(self):
        self.dialog.configure(bg=DarkTheme.BG_PRIMARY)
        main_frame = tk.Frame(self.dialog, bg=DarkTheme.BG_PRIMARY)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        title_label = tk.Label(main_frame, text="Configure Camera Sources", font=("Arial", 14, "bold"), bg=DarkTheme.BG_PRIMARY, fg=DarkTheme.FG_PRIMARY)
        title_label.pack(pady=(0, 15))
        info_label = tk.Label(main_frame, text="Enable cameras and configure their sources. USB cameras use numbers (0,1,2...)", font=("Arial", 10), bg=DarkTheme.BG_PRIMARY, fg=DarkTheme.FG_SECONDARY)
        info_label.pack(pady=(0, 10))
        cameras_frame = tk.Frame(main_frame, bg=DarkTheme.BG_PRIMARY)
        cameras_frame.pack(fill=tk.BOTH, expand=True)
 
        self.create_camera_config('priority_1', 'Priority Camera 1', cameras_frame, 0)
        self.create_camera_config('priority_2', 'Priority Camera 2', cameras_frame, 1)
        self.create_camera_config('object_1', 'Object Camera 1', cameras_frame, 2)
        self.create_camera_config('object_2', 'Object Camera 2', cameras_frame, 3)
        self.create_camera_config('object_3', 'Object Camera 3', cameras_frame, 4)

        button_frame = tk.Frame(main_frame, bg=DarkTheme.BG_PRIMARY)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        start_btn = tk.Button(button_frame, text="Start Selected Cameras", command=self.start_cameras, font=('Arial', 11, 'bold'), bg=DarkTheme.ACCENT, fg=DarkTheme.FG_PRIMARY, activebackground=DarkTheme.ACCENT_HOVER, activeforeground=DarkTheme.FG_PRIMARY, bd=0, padx=20, pady=10)
        start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=self.dialog.destroy, font=('Arial', 11), bg=DarkTheme.BG_TERTIARY, fg=DarkTheme.FG_PRIMARY, activebackground=DarkTheme.BG_SECONDARY, activeforeground=DarkTheme.FG_PRIMARY, bd=0, padx=20, pady=10)
        cancel_btn.pack(side=tk.RIGHT)
    
    def create_camera_config(self, camera_id, display_name, parent, index):
        frame = tk.LabelFrame(parent, text=display_name, bg=DarkTheme.BG_SECONDARY, fg=DarkTheme.FG_PRIMARY, font=('Arial', 10, 'bold'))
        frame.pack(fill=tk.X, pady=5, padx=5)

        controls_frame = tk.Frame(frame, bg=DarkTheme.BG_SECONDARY)
        controls_frame.pack(fill=tk.X, padx=10, pady=8)
        
        self.camera_configs[camera_id] = {'enabled': tk.BooleanVar(value=index < 1),'source_type': tk.StringVar(value='usb'),'source_path': tk.StringVar(value=str(index))}

        enable_cb = tk.Checkbutton(controls_frame, text="Enable", variable=self.camera_configs[camera_id]['enabled'], bg=DarkTheme.BG_SECONDARY,fg=DarkTheme.FG_PRIMARY, selectcolor=DarkTheme.BG_TERTIARY, activebackground=DarkTheme.BG_SECONDARY,activeforeground=DarkTheme.FG_PRIMARY)
        enable_cb.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(controls_frame, text="Type:", bg=DarkTheme.BG_SECONDARY, fg=DarkTheme.FG_PRIMARY).pack(side=tk.LEFT, padx=(0, 5))
        
        type_combo = ttk.Combobox(controls_frame, textvariable=self.camera_configs[camera_id]['source_type'], values=['usb', 'file', 'rtsp', 'http'], width=8, state='readonly')
        type_combo.pack(side=tk.LEFT, padx=(0, 15))

        tk.Label(controls_frame, text="Source:", 
                bg=DarkTheme.BG_SECONDARY, 
                fg=DarkTheme.FG_PRIMARY).pack(side=tk.LEFT, padx=(0, 5))
        
        source_entry = tk.Entry(controls_frame, textvariable=self.camera_configs[camera_id]['source_path'], width=30, bg=DarkTheme.BG_TERTIARY, fg=DarkTheme.FG_PRIMARY, insertbackground=DarkTheme.FG_PRIMARY)
        source_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Browse button for video file loops (not for production use)
        def browse_file():
            if self.camera_configs[camera_id]['source_type'].get() == 'file':
                filename = filedialog.askopenfilename(
                    title=f"Select video file for {display_name}",
                    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
                )
                if filename:
                    self.camera_configs[camera_id]['source_path'].set(filename)
        
        browse_btn = tk.Button(controls_frame, text="Browse", 
                            command=browse_file, width=8,
                            bg=DarkTheme.BG_TERTIARY,
                            fg=DarkTheme.FG_PRIMARY,
                            activebackground=DarkTheme.BG_PRIMARY,
                            activeforeground=DarkTheme.FG_PRIMARY,
                            bd=0, padx=10)
        browse_btn.pack(side=tk.LEFT)
    
    def start_cameras(self): 
        self.camera_manager.stop_all_cameras()
        self.camera_manager.cameras.clear()
        
        for camera_id, config in self.camera_configs.items():
            if config['enabled'].get():
                source_type = config['source_type'].get()
                source_path = config['source_path'].get()
                
                if source_path.strip(): 
                    role = 'priority' if camera_id.startswith('priority') else 'object'
                    camera_config = {'loop': True} if source_type == 'file' else {}
                    
                    self.camera_manager.add_camera(camera_id, source_type, source_path, role, camera_config)
        
        success_count = self.camera_manager.start_all_cameras()
        
        if success_count > 0:
            messagebox.showinfo("Success", f"Started {success_count} cameras")
            self.dialog.destroy()
        else:
            messagebox.showerror("Error", "Failed to start any cameras")
