from flask import Flask, render_template, request
import whisper
import tempfile
import os
import torch
import numpy as np
import requests
from tqdm import tqdm
from transformers import BertTokenizer
from model.multi_class_model import MultiClassModel  # Adjust if needed
import lightning as L

app = Flask(__name__)

# === CONFIG ===
CHECKPOINT_URL = "https://github.com/michael2002porto/bert_classification_indonesian_song_lyrics/releases/download/finetuned_checkpoints/original_split_synthesized.ckpt"
CHECKPOINT_PATH = "final_checkpoint/original_split_synthesized.ckpt"
AGE_LABELS = ["semua usia", "anak", "remaja", "dewasa"]

# === FUNCTION TO DOWNLOAD CKPT IF NEEDED ===
def download_checkpoint_if_needed(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"📥 Downloading model checkpoint from {url}...")
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            total = int(response.headers.get("content-length", 0))
            with open(save_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print("✅ Checkpoint downloaded!")
        else:
            raise Exception(f"❌ Failed to download: {response.status_code}")

# === INITIAL SETUP: Download & Load Model ===
download_checkpoint_if_needed(CHECKPOINT_URL, CHECKPOINT_PATH)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

# Load model from checkpoint
model = MultiClassModel.load_from_checkpoint(
    CHECKPOINT_PATH,
    n_out=4,
    dropout=0.3,
    lr=1e-5
)
model.eval()


# === ROUTES ===

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Load Whisper with Indonesian language support (large / turbo)
        # https://github.com/openai/whisper
        whisper_model = whisper.load_model("large")

        audio_file = request.files['file']
        if audio_file:
            # Save uploaded audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_file.read())
                temp_audio_path = temp_audio.name

            # Step 1: Transcribe
            transcription = whisper_model.transcribe(temp_audio_path, language="id")
            os.remove(temp_audio_path)
            transcribed_text = transcription["text"]

            # Step 2: BERT Prediction
            encoding = tokenizer.encode_plus(
                transcribed_text,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors='pt',
            )

            with torch.no_grad():
                prediction = model(
                    encoding["input_ids"],
                    encoding["attention_mask"],
                    encoding["token_type_ids"]
                )

            logits = prediction
            probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()
            predicted_class = np.argmax(probabilities)
            predicted_label = AGE_LABELS[predicted_class]

            prob_results = [(label, f"{prob:.4f}") for label, prob in zip(AGE_LABELS, probabilities)]

            return render_template(
                'transcribe.html',
                task=transcribed_text,
                prediction=predicted_label,
                probabilities=prob_results
            )

    except Exception as e:
        print("Error:", e)
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
