# Gui styles for a dark theme using tkinter and ttk
import tkinter as tk
from tkinter import ttk

class DarkTheme:
    # Colors
    BG_PRIMARY = '#1e1e1e'
    BG_SECONDARY = '#2d2d2d'
    BG_TERTIARY = '#3d3d3d'
    FG_PRIMARY = '#ffffff'
    FG_SECONDARY = '#b0b0b0'
    FG_TERTIARY = '#808080'
    ACCENT = '#0d7377'
    ACCENT_HOVER = '#14a085'

    
    @staticmethod
    def configure_styles(root=None):
        style = ttk.Style()

        style.configure('Dark.TFrame', background=DarkTheme.BG_PRIMARY)
        style.configure('Dark.TLabel', background=DarkTheme.BG_PRIMARY, 
                       foreground=DarkTheme.FG_PRIMARY)
        style.configure('DarkTitle.TLabel', background=DarkTheme.BG_PRIMARY, 
                       foreground=DarkTheme.FG_PRIMARY, font=('Arial', 14, 'bold'))
        style.configure('DarkInfo.TLabel', background=DarkTheme.BG_PRIMARY, 
                       foreground=DarkTheme.FG_SECONDARY, font=('Arial', 10))
        
        # Label frame
        style.configure('Dark.TLabelframe', background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.FG_PRIMARY, bordercolor=DarkTheme.BG_TERTIARY)
        style.configure('Dark.TLabelframe.Label', background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.FG_PRIMARY)
        
        # Buttons
        style.configure('Dark.TButton', background=DarkTheme.BG_TERTIARY,
                       foreground=DarkTheme.FG_PRIMARY, borderwidth=0)
        style.map('Dark.TButton', background=[('active', DarkTheme.BG_SECONDARY)])
        
        style.configure('Accent.TButton', background=DarkTheme.ACCENT,
                       foreground=DarkTheme.FG_PRIMARY, borderwidth=0)
        style.map('Accent.TButton', background=[('active', DarkTheme.ACCENT_HOVER)])
        
        # Entry
        style.configure('Dark.TEntry', fieldbackground=DarkTheme.BG_TERTIARY,
                       foreground=DarkTheme.FG_PRIMARY, borderwidth=1,
                       insertcolor=DarkTheme.FG_PRIMARY)
        
        # Combo box
        style.configure('Dark.TCombobox', fieldbackground=DarkTheme.BG_TERTIARY,
                       background=DarkTheme.BG_TERTIARY, foreground=DarkTheme.FG_PRIMARY,
                       arrowcolor=DarkTheme.FG_PRIMARY)
        
        # Check button
        style.configure('Dark.TCheckbutton', background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.FG_PRIMARY)
        
        # Menu button
        style.configure('Dark.TMenubutton', background=DarkTheme.BG_TERTIARY,
                       foreground=DarkTheme.FG_PRIMARY, borderwidth=1,
                       relief='flat')
        
        # Entry focus mapping
        style.map('Dark.TEntry', 
                 fieldbackground=[('focus', DarkTheme.BG_TERTIARY)],
                 foreground=[('focus', DarkTheme.FG_PRIMARY)])
        
        if root:
            root.option_add('*TCombobox*Listbox.background', DarkTheme.BG_TERTIARY)
            root.option_add('*TCombobox*Listbox.foreground', DarkTheme.FG_PRIMARY)
            root.option_add('*TCombobox*Listbox.selectBackground', DarkTheme.ACCENT)
            root.option_add('*TCombobox*Listbox.selectForeground', DarkTheme.FG_PRIMARY)

        
