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
import wave

STATE_FILE = "state.json"

class VideoInstallationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Installation Controller")
        # Dark mode colors
        self.bg_color = "#2e2e2e"
        self.fg_color = "#ffffff"
        self.button_bg = "#444444"
        self.button_fg = "#ffffff"
        
        # Control window is managed normally (not full-screen here)
        self.root.configure(bg=self.bg_color)
        
        self.video_path = None
        self.selected_audio_device = 0
        self.fullscreen = False
        self.ambient_noise = 0.0  # Baseline ambient noise level (calibrated)

        # Nuova variabile per file audio e modalità file audio
        self.audio_file = None  
        self.use_audio_file = tk.BooleanVar(value=False)

        # Setup frame (for basic configuration)
        setup_frame = tk.Frame(root, bg=self.bg_color)
        setup_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        tk.Label(setup_frame, text="Setup", bg=self.bg_color, fg=self.fg_color).pack()
        
        self.audio_devices = self.get_audio_devices()
        self.audio_var = tk.StringVar(value=self.audio_devices[0] if self.audio_devices else "Nessun dispositivo")
        self.selected_audio_device = -1 if not self.audio_devices else 0
        
        self.audio_device_menu = tk.OptionMenu(setup_frame, self.audio_var, *self.audio_devices, command=self.select_audio_device)
        self.audio_device_menu.config(bg=self.button_bg, fg=self.button_fg)
        self.audio_device_menu.pack()
        
        tk.Button(setup_frame, text="Test Audio", command=self.test_audio_input, bg=self.button_bg, fg=self.button_fg).pack(pady=2)
        # New button to calibrate ambient noise (in a moment of silence)
        tk.Button(setup_frame, text="Calibra Sala", command=self.calibrate_ambient_noise, bg=self.button_bg, fg=self.button_fg).pack(pady=2)
        self.indicator_label = tk.Label(setup_frame, width=2, height=1, bg='red')
        self.indicator_label.pack()
        
        tk.Button(setup_frame, text="Seleziona Video", command=self.load_video, bg=self.button_bg, fg=self.button_fg).pack(pady=2)
        
        self.screens = [monitor.name for monitor in screeninfo.get_monitors()]
        self.screen_var = tk.StringVar(value=self.screens[0] if self.screens else "Default")
        self.screen_menu = tk.OptionMenu(setup_frame, self.screen_var, *self.screens)
        self.screen_menu.config(bg=self.button_bg, fg=self.button_fg)
        self.screen_menu.pack()
        
        tk.Button(setup_frame, text="Avvia Proiezione", command=self.start_projection, bg=self.button_bg, fg=self.button_fg).pack(pady=2)
        tk.Button(setup_frame, text="Spegni Proiezione", command=self.stop_projection, bg=self.button_bg, fg=self.button_fg).pack(pady=2)
        tk.Button(setup_frame, text="Schermo Intero", command=self.toggle_fullscreen, bg=self.button_bg, fg=self.button_fg).pack(pady=2)
        
        # Audio control frame
        audio_control_frame = tk.Frame(root, bg=self.bg_color)
        audio_control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        tk.Label(audio_control_frame, text="Audio Control", bg=self.bg_color, fg=self.fg_color).pack()
        
        self.min_volume = tk.DoubleVar(value=0.0)   # Minimum volume threshold; will be used together with ambient noise
        self.max_volume = tk.DoubleVar(value=5000.0)  # Maximum volume threshold
        
        tk.Label(audio_control_frame, text="Min Volume", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.0, to=5000.0, resolution=100.0, orient=tk.HORIZONTAL, variable=self.min_volume,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        tk.Label(audio_control_frame, text="Max Volume", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.0, to=5000.0, resolution=100.0, orient=tk.HORIZONTAL, variable=self.max_volume,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()

        # Checkbox per usare file audio
        tk.Checkbutton(audio_control_frame, text="Usa Audio File", variable=self.use_audio_file,
                       bg=self.bg_color, fg=self.fg_color, selectcolor=self.button_bg).pack(pady=2)
        # Pulsante per selezionare il file audio
        tk.Button(audio_control_frame, text="Seleziona Audio File", command=self.load_audio_file, bg=self.button_bg, fg=self.button_fg).pack(pady=2)
        
        # Frequency control sliders (These factors multiply computed frequency values)
        self.low_freq = tk.DoubleVar(value=1.0)   # Factor for low frequency effect
        self.mid_freq = tk.DoubleVar(value=1.0)   # Factor for mid frequency effect (translation effect)
        self.high_freq = tk.DoubleVar(value=1.0)  # Factor for high frequency effect
        
        tk.Label(audio_control_frame, text="Low Frequency", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.low_freq,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        tk.Label(audio_control_frame, text="Mid Frequency", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.mid_freq,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        tk.Label(audio_control_frame, text="High Frequency", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(audio_control_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.high_freq,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        # Effect control frame (for properties affecting visual effects)
        effect_control_frame = tk.Frame(root, bg=self.bg_color)
        effect_control_frame.grid(row=0, column=2, padx=10, pady=10, sticky="n")
        
        tk.Label(effect_control_frame, text="Effect Control", bg=self.bg_color, fg=self.fg_color).pack()
        
        self.min_zoom = tk.DoubleVar(value=1.0)   # Minimum zoom scale factor (default: no zoom)
        self.max_zoom = tk.DoubleVar(value=2.0)   # Maximum zoom scale factor
        self.zoom_speed = tk.DoubleVar(value=0.1)   # Responsiveness of zoom effect
        
        tk.Label(effect_control_frame, text="Min Zoom", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=1.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.min_zoom,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        tk.Label(effect_control_frame, text="Max Zoom", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=1.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.max_zoom,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        tk.Label(effect_control_frame, text="Zoom Speed", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.zoom_speed,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        # Opacity control sliders (for fade effects based on audio volume)
        self.min_opacity = tk.DoubleVar(value=0.0)  # Minimum opacity (fully transparent)
        self.max_opacity = tk.DoubleVar(value=1.0)  # Maximum opacity (fully opaque)
        
        tk.Label(effect_control_frame, text="Min Opacity", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.min_opacity,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        tk.Label(effect_control_frame, text="Max Opacity", bg=self.bg_color, fg=self.fg_color).pack()
        tk.Scale(effect_control_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=self.max_opacity,
                 bg=self.bg_color, fg=self.fg_color, troughcolor=self.button_bg).pack()
        
        # Checkbox control frame (toggles various effects)
        checkbox_control_frame = tk.Frame(root, bg=self.bg_color)
        checkbox_control_frame.grid(row=0, column=3, padx=10, pady=10, sticky="n")
        tk.Label(checkbox_control_frame, text="Effect Toggles", bg=self.bg_color, fg=self.fg_color).pack()
        
        self.low_freq_effect_enabled = tk.BooleanVar(value=False)   # Toggle low frequency effect
        self.mid_freq_effect_enabled = tk.BooleanVar(value=False)   # Toggle mid frequency effect (translation)
        self.high_freq_effect_enabled = tk.BooleanVar(value=False)  # Toggle high frequency effect
        self.opacity_effect_enabled = tk.BooleanVar(value=False)    # Toggle opacity effect
        self.zoom_effect_enabled = tk.BooleanVar(value=False)       # Toggle zoom effect
        self.bypass_effects = tk.BooleanVar(value=False)            # Bypass all effects
        
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
        # New Performance Mode checkbox: disables frequency analysis & visualization
        self.performance_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(checkbox_control_frame, text="Performance Mode", variable=self.performance_mode,
                       bg=self.bg_color, fg=self.fg_color, selectcolor=self.button_bg).pack(pady=5)
        # Save state button
        tk.Button(checkbox_control_frame, text="Save State", command=self.save_state, bg=self.button_bg, fg=self.button_fg).pack(pady=5)
        
        # New visualization layout: Volume indicator and three frequency plots in a row.
        visualization_frame = tk.Frame(root, bg=self.bg_color)
        visualization_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        # Volume indicator
        self.volume_indicator = tk.Label(visualization_frame, text="Volume: 0", bg="gray", fg=self.fg_color, width=10)
        self.volume_indicator.pack(side=tk.LEFT, padx=5)

        # Create matplotlib figure for three frequency subplots arranged horizontally.
        # Each plot ~150 pixels wide (assuming DPI=100, width=1.5 inch each, total 4.5 inches)
        self.fig, axs = plt.subplots(1, 3, figsize=(4.5, 2), dpi=100)
        self.ax_low_vis = axs[0]
        self.ax_mid_vis = axs[1]
        self.ax_high_vis = axs[2]

        # Setup each axis.
        for ax in axs:
            ax.set_ylim(0, 5000)
            ax.set_xlim(0, 1024)
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax_low_vis.set_title("Low Freq")
        self.ax_mid_vis.set_title("Mid Freq")
        self.ax_high_vis.set_title("High Freq")

        self.line_low, = self.ax_low_vis.plot([], [], lw=2)
        self.line_mid, = self.ax_mid_vis.plot([], [], lw=2)
        self.line_high, = self.ax_high_vis.plot([], [], lw=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=visualization_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, padx=5)
        
        # If a state file exists, load it.
        if os.path.exists(STATE_FILE):
            self.load_state(STATE_FILE)

    # ====================================
    # Utility/Interface Functions
    # ====================================
    def load_video(self):
        """Open a file dialog to choose a video file."""
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        print(f"Video selezionato: {self.video_path}")

    def select_audio_device(self, selected_device):
        """Set the selected audio input device based on menu choice."""
        if selected_device in self.audio_devices:
            self.selected_audio_device = self.audio_devices.index(selected_device)
        print(f"Dispositivo audio selezionato: {selected_device} (Index: {self.selected_audio_device})")

    def get_audio_devices(self):
        """Retrieve a list of available audio input devices."""
        p = pyaudio.PyAudio()
        devices = [p.get_device_info_by_index(i)['name'] for i in range(p.get_device_count()) 
                   if p.get_device_info_by_index(i)['maxInputChannels'] > 0]
        p.terminate()
        return devices
    
    def load_audio_file(self):
            """Permette di selezionare un file audio (ad esempio .wav) da usare per gli effetti video."""
            self.audio_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
            print(f"File audio selezionato: {self.audio_file}")

    def test_audio_input(self):
        """Test the current audio input and update the indicator color based on volume."""
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

    def calibrate_ambient_noise(self):
        """
        Calibrate the ambient noise in the room by reading audio data in silence.
        Sets self.ambient_noise and updates the minimum volume threshold.
        """
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, 
                            input_device_index=self.selected_audio_device, frames_per_buffer=1024)
            data = stream.read(2048, exception_on_overflow=False)
            ambient = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
            self.ambient_noise = ambient
            print(f"Ambient noise calibrato: {ambient}")
            # Optionally update the min_volume control according to ambient noise.
            self.min_volume.set(ambient)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Errore nella calibrazione: {e}")
        finally:
            p.terminate()

    def start_projection(self):
        """Start video projection and related threads for audio processing."""
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
        """Read and process video frames in a loop."""
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
        """Legge continuamente dati audio, da file se selezionato oppure dal microfono,
        per aggiornare il volume ed applicare gli effetti video."""
        # Se la modalità file audio è attivata e un file è stato selezionato, usalo
        if self.use_audio_file.get() and self.audio_file:
            ##import wave
            wf = wave.open(self.audio_file, 'rb')
            sample_rate = wf.getframerate()  # frequenza in Hz per calcolare il tempo
            while self.running:
                data = wf.readframes(1024)
                if len(data) == 0:  # Fine file, riparte dall'inizio
                    print("Fine file audio, riavvio...")
                    wf.rewind()
                    continue
                # Calcola il tempo corrente in secondi usando wf.tell()
                current_time = wf.tell() / sample_rate
                # Calcola il volume generale
                self.volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                # Stampa il tempo di riproduzione e il volume
                print(f"Tempo riproduzione: {current_time:.2f} s, Volume: {self.volume:.0f}")
                effective_volume = max(self.volume - self.ambient_noise, 0)
                self.update_audio_indicator(effective_volume)
                self.apply_frequency_effects(data)
                if not self.performance_mode.get():
                    self.update_audio_visualization(data)
            wf.close()
        else:
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
                    effective_volume = max(self.volume - self.ambient_noise, 0)
                    self.update_audio_indicator(effective_volume)
                    self.apply_frequency_effects(data)
                    if not self.performance_mode.get():
                        self.update_audio_visualization(data)
                except Exception as e:
                    print(f"Errore audio: {e}")
                    break
            stream.stop_stream()
            stream.close()
            p.terminate()

    # ====================================
    # Visual Effects Functions
    # ====================================
    def apply_audio_effects(self, frame):
        if self.bypass_effects.get():
            return frame
        if hasattr(self, 'volume'):
            # Se abilitato, applica l'effetto opacità con i valori degli slider min_opacity e max_opacity
            if self.opacity_effect_enabled.get():
                frame = self.apply_opacity_effect(frame)
            # Se abilitato, applica l'effetto zoom utilizzando i controlli max_zoom e zoom_speed
            if self.zoom_effect_enabled.get():
                frame = self.apply_zoom_effect(frame)
        # Applica gli effetti basati sulle frequenze se abilitati
        if self.low_freq_effect_enabled.get():
            frame = self.apply_low_freq_effect(frame)
        if self.mid_freq_effect_enabled.get():
            frame = self.apply_mid_freq_effect(frame)
        if self.high_freq_effect_enabled.get():
            frame = self.apply_high_freq_effect(frame)
        return frame

    def apply_low_freq_effect(self, frame):
        """
        Apply low frequency brightness effect.
        Brightness is increased based on the average low frequency power.
        """
        if hasattr(self, 'low_freq_effect'):
            brightness = 1 + (self.low_freq_effect / 2000.0)##mod from 1000
            frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)
        return frame

    def apply_mid_freq_effect(self, frame):
        """
        Apply mid frequency translation effect.
        Instead of blurring, this shifts the frame horizontally based on the computed mid frequency power.
        """
        if hasattr(self, 'mid_freq_effect'):
            # Compute translation amount from mid frequency effect.
            # The factor 10 is arbitrary; adjust it for stronger/weaker translation.
            shift = int(self.mid_freq_effect * self.mid_freq.get() / 5)##mod from 10
            h, w, _ = frame.shape
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            frame = cv2.warpAffine(frame, M, (w, h))
        return frame

    def apply_high_freq_effect(self, frame):
        """
        Apply high frequency noise effect.
        Noise is added based on the average high frequency power.
        """
        if hasattr(self, 'high_freq_effect'):
            noise_amount = self.high_freq_effect / 1000.0
            noise = np.random.normal(0, noise_amount, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise)
        return frame

    def apply_opacity_effect(self, frame):
        """
        Applica l'effetto di opacità: il frame viene fuso con un frame nero in base al volume,
        passando da min_opacity a max_opacity.
        """
        effective_volume = max(self.volume - self.ambient_noise, 0)
        min_vol = self.min_volume.get()
        max_vol = self.max_volume.get()
        # Mapping lineare del volume (clippato tra 0 e 1)
        scale = np.clip((effective_volume - min_vol) / (max_vol - min_vol), 0, 1)
        min_op = self.min_opacity.get()
        max_op = self.max_opacity.get()
        opacity = min_op + (max_op - min_op) * scale
        if not hasattr(self, 'black_frame') or self.black_frame.shape != frame.shape:
            self.black_frame = np.zeros_like(frame)
        frame = cv2.addWeighted(frame, opacity, self.black_frame, 1 - opacity, 0)
        return frame

    def apply_zoom_effect(self, frame):
        """
        Applica l'effetto di zoom facendo pulsare il video con il suono.
        Il livello di zoom è 1 (zoom normale) quando il volume corrisponde al rumore ambientale,
        aumentando man mano che il volume supera il rumore.
        Il passaggio verso il target viene reso fluido grazie a una velocità regolabile dall'interfaccia.
        """
        effective_volume = max(self.volume - self.ambient_noise, 0)
        # Imposta un valore soglia che determina quanto il volume influenza lo zoom
        threshold = 300.0  # Regola questo valore per modificare la sensibilità dello zoom
        target_zoom = 1.0 + (effective_volume / threshold)
        # Limita lo zoom massimo in base allo slider
        target_zoom = min(target_zoom, self.max_zoom.get())
        # Smoothing: se non esiste current_zoom lo inizializza a 1.0
        if not hasattr(self, 'current_zoom'):
            self.current_zoom = 1.0
        # Aggiorna gradualmente lo zoom in base alla velocità impostata
        zoom_speed = self.zoom_speed.get()  # da slider: valori da lento a veloce
        self.current_zoom += zoom_speed * (target_zoom - self.current_zoom)
        scale = self.current_zoom

        # Applica lo zoom: ridimensiona e ritaglia centralmente il frame
        h, w, _ = frame.shape
        new_h, new_w = int(h * scale), int(w * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        center_h, center_w = new_h // 2, new_w // 2
        y1 = max(0, center_h - h // 2)
        y2 = y1 + h
        x1 = max(0, center_w - w // 2)
        x2 = x1 + w
        # Se il frame ridimensionato è sufficientemente grande, ritaglia centralmente
        if frame_resized.shape[0] >= h and frame_resized.shape[1] >= w:
            frame = frame_resized[y1:y2, x1:x2]
        return frame

    def update_audio_indicator(self, volume):
        """Update the volume indicator label text based on effective volume."""
        self.volume_indicator.config(text=f"Volume: {volume:.0f}")

    def update_audio_visualization(self, data):
        """
        Update frequency visualization plots for audio data.
        Runs only if performance mode is enabled.
        """
        audio_data = np.frombuffer(data, dtype=np.int16)
        self.line_low.set_ydata(audio_data[:256])
        self.line_low.set_xdata(np.arange(256))
        self.canvas.draw()

    def apply_frequency_effects(self, data):
        """
        Compute FFT of audio data and derive frequency effects for low, mid, and high ranges.
        These computed values are stored for later use in visual effects.
        """
        audio_data = np.frombuffer(data, dtype=np.int16)
        fft_data = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(fft_data))
        # Compute average power in frequency bands.
        low_freq_data = np.abs(fft_data[(freqs >= 0) & (freqs < 0.1)])
        mid_freq_data = np.abs(fft_data[(freqs >= 0.1) & (freqs < 0.5)])
        high_freq_data = np.abs(fft_data[(freqs >= 0.5) & (freqs < 1.0)])
        self.low_freq_effect = low_freq_data.mean() * self.low_freq.get() if low_freq_data.size > 0 else 0
        self.mid_freq_effect = mid_freq_data.mean() * self.mid_freq.get() if mid_freq_data.size > 0 else 0
        self.high_freq_effect = high_freq_data.mean() * self.high_freq.get() if high_freq_data.size > 0 else 0
        # Update frequency plots
        self.line_low.set_ydata(low_freq_data if low_freq_data.size > 0 else np.zeros(256))
        self.line_low.set_xdata(np.arange(len(low_freq_data)) if low_freq_data.size > 0 else np.zeros(256))
        self.line_mid.set_ydata(mid_freq_data if mid_freq_data.size > 0 else np.zeros(256))
        self.line_mid.set_xdata(np.arange(len(mid_freq_data)) if mid_freq_data.size > 0 else np.zeros(256))
        self.line_high.set_ydata(high_freq_data if high_freq_data.size > 0 else np.zeros(256))
        self.line_high.set_xdata(np.arange(len(high_freq_data)) if high_freq_data.size > 0 else np.zeros(256))
        self.canvas.draw()

    def stop_projection(self):
        """Stop video projection and release video resources."""
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Proiezione terminata.")

    def toggle_fullscreen(self):
        """
        Toggle the OpenCV window 'Proiezione Video' between fullscreen (frameless) 
        and windowed mode.
        """
        self.fullscreen = not self.fullscreen
        cv2.destroyWindow("Proiezione Video")
        cv2.namedWindow("Proiezione Video", cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty("Proiezione Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Proiezione Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    def save_state(self, file_path=STATE_FILE):
        """Save current configuration to a JSON file."""
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
            "screen": self.screen_var.get(),
            "audio_file": self.audio_file,
            "full_screen": self.fullscreen,
            "performance_mode": self.performance_mode.get(),
            "projection_running": getattr(self, 'running', False)
        }
        with open(file_path, "w") as f:
            json.dump(state, f)
        print("State saved to", file_path)
    
    def load_state(self, file_path=STATE_FILE):
        """Load configuration from a JSON file."""
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
            self.audio_file = state.get("audio_file", None)
            # Imposta la checkbox in base alla presenza di file audio
            self.use_audio_file.set(bool(self.audio_file))
            self.fullscreen = state.get("full_screen", False)
            self.performance_mode.set(state.get("performance_mode", False))
            print("State loaded from", file_path)
            
            # Se la proiezione era attiva nel salvataggio e un video è stato selezionato, avvia la proiezione
            if state.get("projection_running", False) and self.video_path:
                print("Riaccendendo la proiezione salvata...")
                self.start_projection()
                # Se era in fullscreen, attiva il tutto schermo
                if self.fullscreen:
                    self.toggle_fullscreen()
        except Exception as e:
            print("Failed to load state:", e)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoInstallationApp(root)
    root.mainloop()