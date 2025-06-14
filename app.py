import os
import gradio as gr
from faster_whisper import WhisperModel
from datetime import datetime
import tempfile

model = WhisperModel("base", compute_type="int8")

def transcribe(audio):
    segments, _ = model.transcribe(audio, beam_size=5)
    full_text = ""
    for segment in segments:
        full_text += f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}\n"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(tempfile.gettempdir(), f"transcript_{timestamp}.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    return full_text, save_path

with gr.Blocks() as demo:
    gr.Markdown("## 🎤 BeMyEars Lite (Web Version)")
    audio_input = gr.Audio(source="microphone", type="filepath", label="Speak English or Chinese")
    transcribe_btn = gr.Button("Transcribe")
    output_text = gr.Textbox(label="Transcription")
    download_file = gr.File(label="Download Transcript")

    transcribe_btn.click(fn=transcribe, inputs=audio_input, outputs=[output_text, download_file])

demo.launch()
