import os
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import google.generativeai as genai
import queue
from dotenv import load_dotenv

class AudioSummarizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Summarizer")
        self.master.geometry("400x300")

        self.recording = False
        self.audio_file = "recording.wav"
        self.fs = 44100  # Sample rate
        self.audio_queue = queue.Queue()

        self.setup_gemini()
        self.create_widgets()

    def setup_gemini(self):
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-002",
            generation_config=generation_config,
            system_instruction="You are a world-class summarizer. Create a summary based on the content below."
        )

    def create_widgets(self):
        self.button_record = tk.Button(self.master, text="Start Recording", command=self.toggle_recording)
        self.button_record.pack(pady=10)

        self.button_summarize = tk.Button(self.master, text="Summarize", command=self.summarize_audio, state=tk.DISABLED)
        self.button_summarize.pack(pady=10)

        self.text_summary = tk.Text(self.master, wrap=tk.WORD, height=10)
        self.text_summary.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.button_record.config(text="Stop Recording", bg="red")
        self.button_summarize.config(state=tk.DISABLED)
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            self.audio_queue.put(indata.copy())

        self.stream = sd.InputStream(samplerate=self.fs, channels=1, callback=audio_callback)
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.button_record.config(text="Start Recording", bg="SystemButtonFace")
        self.button_summarize.config(state=tk.NORMAL)
        
        self.stream.stop()
        self.stream.close()

        recorded_data = []
        while not self.audio_queue.empty():
            recorded_data.append(self.audio_queue.get())

        recording = np.concatenate(recorded_data, axis=0)
        wav.write(self.audio_file, self.fs, recording)
        # Removed the messagebox.showinfo call here

    def upload_to_gemini(self, path, mime_type=None):
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def summarize_audio(self):
        try:
            audio_file = self.upload_to_gemini(self.audio_file, mime_type="audio/wav")

            chat_session = self.model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [audio_file],
                    },
                ]
            )

            response = chat_session.send_message("Please provide a summary of the audio content.")

            self.text_summary.delete("1.0", tk.END)
            self.text_summary.insert(tk.END, response.text)

        except Exception as e:
            messagebox.showerror("Error", f"Error during summarization: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioSummarizerApp(root)
    root.mainloop()