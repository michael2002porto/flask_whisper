from flask import Flask, render_template, request
# import whisper
import io
import os
import time
import torch
import torchaudio
import numpy as np
import requests
from tqdm import tqdm
from transformers import BertTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from model.multi_class_model import MultiClassModel  # Adjust if needed

app = Flask(__name__)

# === CONFIG ===
# CHECKPOINT_URL = "https://github.com/michael2002porto/bert_classification_indonesian_song_lyrics/releases/download/finetuned_checkpoints/original_split_synthesized.ckpt"
CHECKPOINT_URL = "https://huggingface.co/nenafem/original_split_synthesized/resolve/main/original_split_synthesized.ckpt?download=true"
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


def whisper_api(input_audio):
    # https://huggingface.co/openai/whisper-large-v3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,  # batch size for inference - set based on your device
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(input_audio, return_timestamps=False, generate_kwargs={"language": "indonesian"})
    print(result["text"])
    return result


# === ROUTES ===

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Load Whisper with Indonesian language support (large / turbo)
        # https://github.com/openai/whisper
        # whisper_model = whisper.load_model("large")

        # Start measuring time
        start_time = time.time()

        audio_file = request.files['file']
        if audio_file:
            # Read uploaded file into memory
            audio_bytes = audio_file.read()
            # Load audio from bytes directly
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
            # Convert to mono if it is stereo
            waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
            # Convert waveform to numpy
            audio_array = waveform.squeeze(0).numpy()

            # Step 1: Transcribe
            # transcription = whisper_model.transcribe(temp_audio_path, language="id")
            transcription = whisper_api({"array": audio_array, "sampling_rate": sample_rate})
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

            # Stop timer
            end_time = time.time()
            total_time = end_time - start_time
            formatted_time = f"{total_time:.2f} seconds"

            return render_template(
                'transcribe.html',
                task=transcribed_text,
                prediction=predicted_label,
                probabilities=prob_results,
                total_time=formatted_time
            )

    except Exception as e:
        print("Error:", e)
        return str(e)


@app.route('/predict-text', methods=['POST'])
def predict_text():
    try:
        user_lyrics = request.form.get('lyrics', '').strip()

        if not user_lyrics:
            return "No lyrics provided.", 400

        # Start timer
        start_time = time.time()

        encoding = tokenizer.encode_plus(
            user_lyrics,
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

        # End timer
        end_time = time.time()
        total_time = f"{end_time - start_time:.2f} seconds"

        return render_template(
            'transcribe.html',
            task=user_lyrics,
            prediction=predicted_label,
            probabilities=prob_results,
            total_time=total_time
        )

    except Exception as e:
        print("❌ Error in predict-text:", e)
        return str(e), 500


if __name__ == "__main__":
    app.run(debug=True)
