import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pyaudio
import threading
import screeninfo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

STATE_FILE = "state.json"

class VideoInstallationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Installation Controller")
        # Set dark mode colors
        self.bg_color = "#2e2e2e"
        self.fg_color = "#ffffff"
        self.button_bg = "#444444"
        self.button_fg = "#ffffff"
        
        # Remove these lines so the control window is managed normally.
        # self.root.overrideredirect(True)
        # self.root.attributes("-fullscreen", True)
        self.root.configure(bg=self.bg_color)
        
        self.video_path = None
        self.selected_audio_device = 0
        self.fullscreen = False

        # Setup frame
        setup_frame = tk.Frame(root, bg=self.bg_color)
        setup_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        
        tk.Label(setup_frame, text="Setup", bg=self.bg_color, fg=self.fg_color).pack()
        
        self.audio_devices = self.get_audio_devices()
        self.audio_var = tk.StringVar(value=self.audio_devices[0] if self.audio_devices else "Nessun dispositivo")
        self.selected_audio_device = -1 if not self.audio_devices else 0
        
        self.audio_device_menu = tk.OptionMenu(setup_frame, self.audio_var, *self.audio_devices, command=self.select_audio_device)
        self.audio_device_menu.config(bg=self.button_bg, fg=self.button_fg)
        self.audio_device_menu.pack()
        
        tk.Button(setup_frame, text="Test Audio", command=self.test_audio_input, bg=self.button_bg, fg=self.button_fg).pack()
        self.indicator_label = tk.Label(setup_frame, width=2, height=1, bg='red')
        self.indicator_label.pack()
        
        tk.Button(setup_frame, text="Seleziona Video", command=self.load_video, bg=self.button_bg, fg=self.button_fg).pack()
        
        self.screens = [monitor.name for monitor in screeninfo.get_monitors()]
        self.screen_var = tk.StringVar(value=self.screens[0] if self.screens else "Default")
        self.screen_menu = tk.OptionMenu(setup_frame, self.screen_var, *self.screens)
        self.screen_menu.config(bg=self.button_bg, fg=self.button_fg)
        self.screen_menu.pack()
        
        tk.Button(setup_frame, text="Avvia Proiezione", command=self.start_projection, bg=self.button_bg, fg=self.button_fg).pack()
        tk.Button(setup_frame, text="Spegni Proiezione", command=self.stop_projection, bg=self.button_bg, fg=self.button_fg).pack()
        tk.Button(setup_frame, text="Schermo Intero", command=self.toggle_fullscreen, bg=self.button_bg, fg=self.button_fg).pack()
        
        # Audio control frame
        audio_control_frame = tk.Frame(root, bg=self.bg_color)
        audio_control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        
        tk.Label(audio_control_frame, text="Audio Control", bg=self.bg_color, fg=self.fg_color).pack()
        
        self.min_volume = tk.DoubleVar(value=0.0)
        self.max_volume = tk.DoubleVar(value=5000.0)
        
        tk.Label(audio_control_frame, text="Min Volume", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.0, to=5000.0, resolution=100.0, orient=tk.HORIZONTAL, variable=self.min_volume,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        tk.Label(audio_control_frame, text="Max Volume", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.0, to=5000.0, resolution=100.0, orient=tk.HORIZONTAL, variable=self.max_volume,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        # Add frequency control sliders
        self.low_freq = tk.DoubleVar(value=1.0)
        self.mid_freq = tk.DoubleVar(value=1.0)
        self.high_freq = tk.DoubleVar(value=1.0)
        
        tk.Label(audio_control_frame, text="Low Frequency", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.low_freq,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        tk.Label(audio_control_frame, text="Mid Frequency", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.mid_freq,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        tk.Label(audio_control_frame, text="High Frequency", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.high_freq,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        # Effect control frame
        effect_control_frame = tk.Frame(root, bg=self.bg_color)
        effect_control_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
        
        tk.Label(effect_control_frame, text="Effect Control", bg=self.bg_color, fg=self.fg_color).pack()
        
        self.min_zoom = tk.DoubleVar(value=1.0)
        self.max_zoom = tk.DoubleVar(value=2.0)
        self.zoom_speed = tk.DoubleVar(value=0.1)
        
        tk.Label(effect_control_frame, text="Min Zoom", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=1.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.min_zoom,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        tk.Label(effect_control_frame, text="Max Zoom", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=1.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.max_zoom,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        tk.Label(effect_control_frame, text="Zoom Speed", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.zoom_speed,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        # Sliders for opacity control
        self.min_opacity = tk.DoubleVar(value=0.0)
        self.max_opacity = tk.DoubleVar(value=1.0)
        
        tk.Label(effect_control_frame, text="Min Opacity", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.min_opacity,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        tk.Label(effect_control_frame, text="Max Opacity", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.max_opacity,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        # Checkbox control frame (all effects disabled by default)
        checkbox_control_frame = tk.Frame(root, bg=self.bg_color)
        checkbox_control_frame.grid(row=0, column=3, padx=10, pady=10, sticky="n")
        
        tk.Label(checkbox_control_frame, text="Effect Toggles", bg=self.bg_color, fg=self.fg_color).pack()
        
        self.low_freq_effect_enabled = tk.BooleanVar(value=False)
        self.mid_freq_effect_enabled = tk.BooleanVar(value=False)
        self.high_freq_effect_enabled = tk.BooleanVar(value=False)
        self.opacity_effect_enabled = tk.BooleanVar(value=False)
        self.zoom_effect_enabled = tk.BooleanVar(value=False)
        self.bypass_effects = tk.BooleanVar(value=False)
        
        tk.Checkbutton(checkbox_control_frame, text="Enable Low Frequency Effect", variable=self.low_freq_effect_enabled,
                       bg=self.bg_color, fg=self.fg_color, selectcolor=self.button_bg).pack()
        tk.Checkbutton(checkbox_control_frame, text="Enable Mid Frequency Effect", variable=self.mid_freq_effect_enabled,
                       bg=self.bg_color, fg=self.fg_color, selectcolor=self.button_bg).pack()
        tk.Checkbutton(checkbox_control_frame, text="Enable High Frequency Effect", variable=self.high_freq_effect_enabled,
                       bg=self.bg_color, fg=self.fg_color, selectcolor=self.button_bg).pack()
        tk.Checkbutton(checkbox_control_frame, text="Enable Opacity Effect", variable=self.opacity_effect_enabled,
                       bg=self.bg_color, fg=self.fg_color, selectcolor=self.button_bg).pack()
        tk.Checkbutton(checkbox_control_frame, text="Enable Zoom Effect", variable=self.zoom_effect_enabled,
                       bg=self.bg_color, fg=self.fg_color, selectcolor=self.button_bg).pack()
        tk.Checkbutton(checkbox_control_frame, text="Bypass All Effects", variable=self.bypass_effects,
                       bg=self.bg_color, fg=self.fg_color, selectcolor=self.button_bg).pack()
        
        # Add a Save State button to the checkbox control frame
        tk.Button(checkbox_control_frame, text="Save State", command=self.save_state, bg=self.button_bg, fg=self.button_fg).pack(pady=5)
        
        # Audio visualization frame with four subplots
        visualization_frame = tk.Frame(root, bg=self.bg_color)
        visualization_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="ew")
        
        self.fig, (self.ax, self.ax_low_vis, self.ax_mid_vis, self.ax_high_vis) = plt.subplots(4, 1, figsize=(5, 8))
        plt.style.use("dark_background")
        self.line, = self.ax.plot([], [], lw=2)
        self.line_low, = self.ax_low_vis.plot([], [], lw=2)
        self.line_mid, = self.ax_mid_vis.plot([], [], lw=2)
        self.line_high, = self.ax_high_vis.plot([], [], lw=2)
        
        self.ax.set_ylim(0, 5000)
        self.ax.set_xlim(0, 512)
        self.ax_low_vis.set_ylim(0, 5000)
        self.ax_low_vis.set_xlim(0, 1024)
        self.ax_mid_vis.set_ylim(0, 5000)
        self.ax_mid_vis.set_xlim(0, 1024)
        self.ax_high_vis.set_ylim(0, 5000)
        self.ax_high_vis.set_xlim(0, 1024)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=visualization_frame)
        self.canvas.get_tk_widget().pack()
        
        # If a state file exists, load it
        if os.path.exists(STATE_FILE):
            self.load_state(STATE_FILE)

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        print(f"Video selezionato: {self.video_path}")

    def select_audio_device(self, selected_device):
        if selected_device in self.audio_devices:
            self.selected_audio_device = self.audio_devices.index(selected_device)
        print(f"Dispositivo audio selezionato: {selected_device} (Index: {self.selected_audio_device})")
    
    def get_audio_devices(self):
        p = pyaudio.PyAudio()
        devices = [p.get_device_info_by_index(i)['name'] for i in range(p.get_device_count()) 
                   if p.get_device_info_by_index(i)['maxInputChannels'] > 0]
        p.terminate()
        return devices
    
    def test_audio_input(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, 
                            input_device_index=self.selected_audio_device, frames_per_buffer=1024)
            data = stream.read(2048, exception_on_overflow=False)
            volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
            print(f"Volume rilevato: {volume}")
            self.indicator_label.config(bg='green' if volume > 50 else 'red')
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Errore audio: {e}")
            self.indicator_label.config(bg='red')
        finally:
            p.terminate()

    def start_projection(self):
        if not self.video_path:
            print("Seleziona prima un video!")
            return
        
        self.running = True
        print("Avviando la proiezione...")

        screen_name = self.screen_var.get()
        print(f"Proiettando su: {screen_name}")

        self.cap = cv2.VideoCapture(self.video_path)
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()
        self.play_video()

    def play_video(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.apply_audio_effects(frame)
            cv2.imshow("Proiezione Video", frame)
            cv2.waitKey(5)
            self.root.after(10, self.play_video)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.play_video()

    def process_audio(self):
        if self.selected_audio_device == -1:
            print("Nessun dispositivo audio disponibile.")
            return
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, 
                        input_device_index=self.selected_audio_device, frames_per_buffer=1024)
        
        while self.running:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                self.volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                self.update_audio_indicator(self.volume)
                self.update_audio_visualization(data)
                self.apply_frequency_effects(data)
            except Exception as e:
                print(f"Errore audio: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()

    def apply_audio_effects(self, frame):
        if self.bypass_effects.get():
            return frame

        if hasattr(self, 'volume'):
            min_vol = self.min_volume.get()
            max_vol = self.max_volume.get()
            min_opacity = self.min_opacity.get()
            max_opacity = self.max_opacity.get()
            
            opacity = min_opacity + (max_opacity - min_opacity) * np.clip((self.volume - min_vol) / (max_vol - min_vol), 0, 1)

            if not hasattr(self, 'black_frame') or self.black_frame.shape != frame.shape:
                self.black_frame = np.zeros_like(frame)

            frame = cv2.addWeighted(frame, opacity, self.black_frame, 1 - opacity, 0)

            if self.zoom_effect_enabled.get():
                min_zoom = self.min_zoom.get()
                max_zoom = self.max_zoom.get()
                scale = min_zoom + (max_zoom - min_zoom) * np.clip((self.volume - min_vol) / (max_vol - min_vol), 0, 1)
                h, w, _ = frame.shape
                new_h, new_w = int(h * scale), int(w * scale)
                resized_frame = cv2.resize(frame, (new_w, new_h))
                center_h, center_w = new_h // 2, new_w // 2
                frame = resized_frame[max(0, center_h - h // 2):min(new_h, center_h + h // 2),
                                       max(0, center_w - w // 2):min(new_w, center_w + w // 2)]
        if self.low_freq_effect_enabled.get():
            frame = self.apply_low_freq_effect(frame)
        if self.mid_freq_effect_enabled.get():
            frame = self.apply_mid_freq_effect(frame)
        if self.high_freq_effect_enabled.get():
            frame = self.apply_high_freq_effect(frame)

        return frame

    def apply_low_freq_effect(self, frame):
        if hasattr(self, 'low_freq_effect'):
            brightness = 1 + (self.low_freq_effect / 1000.0)
            frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
        return frame

    def apply_mid_freq_effect(self, frame):
        if hasattr(self, 'mid_freq_effect'):
            blur_amount = int(self.mid_freq_effect / 1000.0)
            if blur_amount > 0:
                frame = cv2.GaussianBlur(frame, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
        return frame

    def apply_high_freq_effect(self, frame):
        if hasattr(self, 'high_freq_effect'):
            noise_amount = self.high_freq_effect / 1000.0
            noise = np.random.normal(0, noise_amount, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise)
        return frame

    def update_audio_indicator(self, volume):
        self.indicator_label.config(bg='green' if volume > 1000 else 'red')

    def update_audio_visualization(self, data):
        audio_data = np.frombuffer(data, dtype=np.int16)
        self.line.set_ydata(audio_data[:512])
        self.line.set_xdata(np.arange(512))
        self.canvas.draw()

    def apply_frequency_effects(self, data):
        audio_data = np.frombuffer(data, dtype=np.int16)
        fft_data = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft_data))

        low_freq_data = np.abs(fft_data[(freqs >= 0) & (freqs < 0.1)])
        mid_freq_data = np.abs(fft_data[(freqs >= 0.1) & (freqs < 0.5)])
        high_freq_data = np.abs(fft_data[(freqs >= 0.5) & (freqs < 1.0)])

        self.low_freq_effect = low_freq_data.mean() * self.low_freq.get() if low_freq_data.size > 0 else 0
        self.mid_freq_effect = mid_freq_data.mean() * self.mid_freq.get() if mid_freq_data.size > 0 else 0
        self.high_freq_effect = high_freq_data.mean() * self.high_freq.get() if high_freq_data.size > 0 else 0

        self.line_low.set_ydata(low_freq_data if low_freq_data.size > 0 else np.zeros(1024))
        self.line_low.set_xdata(np.arange(len(low_freq_data)) if low_freq_data.size > 0 else np.zeros(1024))
        self.line_mid.set_ydata(mid_freq_data if mid_freq_data.size > 0 else np.zeros(1024))
        self.line_mid.set_xdata(np.arange(len(mid_freq_data)) if mid_freq_data.size > 0 else np.zeros(1024))
        self.line_high.set_ydata(high_freq_data if high_freq_data.size > 0 else np.zeros(1024))
        self.line_high.set_xdata(np.arange(len(high_freq_data)) if high_freq_data.size > 0 else np.zeros(1024))
        self.canvas.draw()

    def stop_projection(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Proiezione terminata.")

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        # First, close the existing window if any.
        cv2.destroyWindow("Proiezione Video")
        # Create a new window in normal mode.
        cv2.namedWindow("Proiezione Video", cv2.WINDOW_NORMAL)
        if self.fullscreen:
            # Set the window to be fullscreen and frameless.
            cv2.setWindowProperty("Proiezione Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Proiezione Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    def save_state(self, file_path=STATE_FILE):
        state = {
            "audio_device": self.audio_var.get(),
            "min_volume": self.min_volume.get(),
            "max_volume": self.max_volume.get(),
            "low_freq": self.low_freq.get(),
            "mid_freq": self.mid_freq.get(),
            "high_freq": self.high_freq.get(),
            "min_zoom": self.min_zoom.get(),
            "max_zoom": self.max_zoom.get(),
            "zoom_speed": self.zoom_speed.get(),
            "min_opacity": self.min_opacity.get(),
            "max_opacity": self.max_opacity.get(),
            "low_freq_effect_enabled": self.low_freq_effect_enabled.get(),
            "mid_freq_effect_enabled": self.mid_freq_effect_enabled.get(),
            "high_freq_effect_enabled": self.high_freq_effect_enabled.get(),
            "opacity_effect_enabled": self.opacity_effect_enabled.get(),
            "zoom_effect_enabled": self.zoom_effect_enabled.get(),
            "bypass_effects": self.bypass_effects.get(),
            "video_path": self.video_path,
            "screen": self.screen_var.get()
        }
        with open(file_path, "w") as f:
            json.dump(state, f)
        print("State saved to", file_path)
    
    def load_state(self, file_path=STATE_FILE):
        try:
            with open(file_path, "r") as f:
                state = json.load(f)
            self.audio_var.set(state.get("audio_device", ""))
            self.min_volume.set(state.get("min_volume", 0.0))
            self.max_volume.set(state.get("max_volume", 5000.0))
            self.low_freq.set(state.get("low_freq", 1.0))
            self.mid_freq.set(state.get("mid_freq", 1.0))
            self.high_freq.set(state.get("high_freq", 1.0))
            self.min_zoom.set(state.get("min_zoom", 1.0))
            self.max_zoom.set(state.get("max_zoom", 2.0))
            self.zoom_speed.set(state.get("zoom_speed", 0.1))
            self.min_opacity.set(state.get("min_opacity", 0.0))
            self.max_opacity.set(state.get("max_opacity", 1.0))
            self.low_freq_effect_enabled.set(state.get("low_freq_effect_enabled", False))
            self.mid_freq_effect_enabled.set(state.get("mid_freq_effect_enabled", False))
            self.high_freq_effect_enabled.set(state.get("high_freq_effect_enabled", False))
            self.opacity_effect_enabled.set(state.get("opacity_effect_enabled", False))
            self.zoom_effect_enabled.set(state.get("zoom_effect_enabled", False))
            self.bypass_effects.set(state.get("bypass_effects", False))
            self.video_path = state.get("video_path", None)
            self.screen_var.set(state.get("screen", ""))
            print("State loaded from", file_path)
        except Exception as e:
            print("Failed to load state:", e)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoInstallationApp(root)
    root.mainloop()