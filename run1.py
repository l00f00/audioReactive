import cv2
import numpy as np
import tkinter as tkiml
from tkinter import filedialog
import pyaudio
import threading
import screeninfo

class VideoInstallationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Installation Controller")
        
        self.video_path = None
        self.selected_audio_device = 0
        self.running = False
        self.fullscreen = False  # Stato fullscreen

        # Dispositivi audio
        self.audio_devices = self.get_audio_devices()
        self.audio_var = tk.StringVar(value=self.audio_devices[0] if self.audio_devices else "Nessun dispositivo")
        self.selected_audio_device = -1 if not self.audio_devices else 0
        
        self.audio_device_menu = tk.OptionMenu(root, self.audio_var, *self.audio_devices, command=self.select_audio_device)
        self.audio_device_menu.pack()
        tk.Button(root, text="Test Audio", command=self.test_audio_input).pack()
        self.indicator_label = tk.Label(root, width=2, height=1, bg='red')
        self.indicator_label.pack()
        
        tk.Button(root, text="Seleziona Video", command=self.load_video).pack()
        
        # Dropdown per selezionare lo schermo
        self.screens = [monitor.name for monitor in screeninfo.get_monitors()]
        self.screen_var = tk.StringVar(value=self.screens[0] if self.screens else "Default")
        self.screen_menu = tk.OptionMenu(root, self.screen_var, *self.screens)
        self.screen_menu.pack()
        
        tk.Button(root, text="Avvia Proiezione", command=self.start_projection).pack()
        tk.Button(root, text="Chiudi Video", command=self.stop_projection).pack()
        
        self.cap = None
        self.audio_thread = None
    
    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        print(f"Video selezionato: {self.video_path}")
    
    def select_audio_device(self, selected_device):
        if selected_device in self.audio_devices:
            self.selected_audio_device = self.audio_devices.index(selected_device)
        print(f"Dispositivo audio selezionato: {selected_device} (Index: {self.selected_audio_device})")
    
    def get_audio_devices(self):
        p = pyaudio.PyAudio()
        devices = [p.get_device_info_by_index(i)['name'] for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxInputChannels'] > 0]
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
            
            # Se attivato fullscreen
            cv2.namedWindow("Proiezione Video", cv2.WND_PROP_FULLSCREEN)
            if self.fullscreen:
                cv2.setWindowProperty("Proiezione Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow("Proiezione Video", frame)
            key = cv2.waitKey(1)
            
            if key == 27:  # ESC chiude il video
                self.stop_projection()
                return
            elif key == 32: 
                self.toggle_fullscreen()
            
            self.root.after(10, self.play_video)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.play_video()
    
    def toggle_fullscreen(self):
        """Attiva/disattiva la modalità fullscreen con F11."""
        self.fullscreen = not self.fullscreen

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
            except Exception as e:
                print(f"Errore audio: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def apply_audio_effects(self, frame):
        if hasattr(self, 'volume'):
            scale = 1 + (self.volume / 500.0)
            h, w, _ = frame.shape
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            center_h, center_w = new_h // 2, new_w // 2
            frame = frame[max(0, center_h - h//2):min(new_h, center_h + h//2),
                          max(0, center_w - w//2):min(new_w, center_w + w//2)]
        return frame
    
    def update_audio_indicator(self, volume):
        self.indicator_label.config(bg='green' if volume > 1000 else 'red')
    
    def stop_projection(self):
        """Ferma la riproduzione del video e chiude la finestra OpenCV."""
        self.running = False
        self.fullscreen = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Proiezione terminata.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoInstallationApp(root)
    root.mainloop()
