from pathlib import Path

import gradio as gr
import librosa
from transformers import (AutoFeatureExtractor,
                          AutoModelForAudioClassification, pipeline)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "sanchit-gandhi/whisper-medium-fleurs-lang-id",
    do_normalize=True
)
SR = 16_000

label2id = {
    'blues': '0',
    'classical': '1',
    'country': '2',
    'disco': '3',
    'hiphop': '4',
    'jazz': '5',
    'metal': '6',
    'pop': '7',
    'reggae': '8',
    'rock': '9'
}
num_labels = len(label2id)

path = "best_Whisper-Small_model_92"
model = AutoModelForAudioClassification.from_pretrained("allispaul/whisper-small-gtzan")
classifier = pipeline("audio-classification", model=model, feature_extractor=feature_extractor)

def predict(audio):
    if isinstance(audio, str) or isinstance(audio, Path):
        audio, sr = librosa.load(audio, sr=16_000)
    else:
        sr, audio = audio
    preds = classifier({"array": audio, "sampling_rate": sr})
    probs = {k: 0.0 for k in label2id}
    for pred in preds:
        probs[pred["label"]] = pred["score"]
    return probs

title = "🤖 🎵 Audiobot 🎹 ⚡"
description = "<h2>♯ The superpowered music genre classifier ♭</h2>"
article = """
<p>🤖 This model is a version of <a href="https://huggingface.co/openai/whisper-small">Whisper Small</a>,
fine-tuned on the <a href="https://huggingface.co/datasets/marsyas/gtzan">GTZAN</a> dataset.</p>
<p>🎹 It recognizes 10 genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock.</p>
<p>🎵 Upload a song or click one of the examples to try it out!</p>
<p>⚡ Part of a project for the <a href="https://www.erdosinstitute.org/">Erdős Institute</a> Deep Learning bootcamp,
by Dylan Bates, Muhammed Cifci, Aycan Katitas, Johann Thiel, Soheyl Anbouhi, and Paul VanKoughnett.</p>
"""
iface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description,
    article=article,
    examples=["examples/country.00008.wav", "examples/hiphop.00057.wav", "examples/gamblersblues.opus"]
)
iface.launch()
