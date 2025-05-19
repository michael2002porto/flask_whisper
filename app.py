from flask import Flask, render_template, request, url_for, redirect
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from faster_whisper import WhisperModel
import tempfile
import os
import time
import torch
import numpy as np
import requests
from tqdm import tqdm
from transformers import BertTokenizer
from model.multi_class_model import MultiClassModel

# from model.database import db, User
from sqlalchemy.exc import OperationalError
from sqlalchemy import inspect

app = Flask(__name__)

# === CONFIG ===
# CHECKPOINT_URL = "https://github.com/michael2002porto/bert_classification_indonesian_song_lyrics/releases/download/finetuned_checkpoints/original_split_synthesized.ckpt"
CHECKPOINT_URL = "https://huggingface.co/nenafem/original_split_synthesized/resolve/main/original_split_synthesized.ckpt?download=true"
CHECKPOINT_PATH = "final_checkpoint/original_split_synthesized.ckpt"
AGE_LABELS = ["semua usia", "anak", "remaja", "dewasa"]
DATABASE_URI = "postgresql://postgres.tcqmmongiztvqkxxebnc:I1Nnj0H72Z3mXWcp@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

# === CONNECT DATABASE ===
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URI
app.config["SECRET_KEY"] = "I1Nnj0H72Z3mXWcp"

# init extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

try:
    db.session.execute("SELECT 1")
    print("✅ Database connected successfully.")
except OperationalError as e:
    print(f"❌ Database connection failed: {e}")


def show_schema_info():
    inspector = inspect(db.engine)

    # Get current schema (by default it's 'public' unless set explicitly)
    current_schema = db.engine.url.database
    all_schemas = inspector.get_schema_names()
    public_tables = inspector.get_table_names(schema="public")

    return {
        "current_schema": current_schema,
        "available_schemas": all_schemas,
        "public_tables": public_tables,
    }


class User(db.Model, UserMixin):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255))  # Can be NULL
    created_date = db.Column(db.DateTime)

    history = db.relationship("History", backref="user", lazy=True)


class History(db.Model):
    __tablename__ = "history"

    id = db.Column(db.Integer, primary_key=True)
    lyric = db.Column(db.String(255), nullable=False)
    predicted_label = db.Column(db.String(255), nullable=False)

    children_prob = db.Column(db.Float)
    adolescents_prob = db.Column(db.Float)
    adults_prob = db.Column(db.Float)
    all_ages_prob = db.Column(db.Float)

    processing_time = db.Column(db.Time)
    created_date = db.Column(db.DateTime)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))


# Load user for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# === FUNCTION TO DOWNLOAD CKPT IF NEEDED ===
def download_checkpoint_if_needed(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"📥 Downloading model checkpoint from {url}...")
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            total = int(response.headers.get("content-length", 0))
            with open(save_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
            print("✅ Checkpoint downloaded!")
        else:
            raise Exception(f"❌ Failed to download: {response.status_code}")


# === INITIAL SETUP: Download & Load Model ===
print(show_schema_info())
download_checkpoint_if_needed(CHECKPOINT_URL, CHECKPOINT_PATH)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("indolem/indobert-base-uncased")

# Load model from checkpoint
model = MultiClassModel.load_from_checkpoint(
    CHECKPOINT_PATH, n_out=4, dropout=0.3, lr=1e-5
)
model.eval()

# === INITIAL SETUP: Faster Whisper ===
# https://github.com/SYSTRAN/faster-whisper
# faster_whisper_model_size = "large-v3"
faster_whisper_model_size = "turbo"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
faster_whisper_model = WhisperModel(
    faster_whisper_model_size, device="cpu", compute_type="int8"
)


def faster_whisper(temp_audio_path):
    segments, info = faster_whisper_model.transcribe(
        temp_audio_path,
        language="id",
        beam_size=1,  # Lower beam_size, faster but may miss words
    )

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    return " ".join(segment.text for segment in segments)


def bert_predict(input_lyric):
    encoding = tokenizer.encode_plus(
        input_lyric,
        add_special_tokens=True,
        max_length=512,
        truncation=True,  # Ensures input ≤512 tokens
        return_token_type_ids=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        prediction = model(
            encoding["input_ids"],
            encoding["attention_mask"],
            encoding["token_type_ids"],
        )

    logits = prediction
    probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()
    predicted_class = np.argmax(probabilities)
    predicted_label = AGE_LABELS[predicted_class]

    prob_results = [
        (label, f"{prob:.4f}") for label, prob in zip(AGE_LABELS, probabilities)
    ]
    return predicted_label, prob_results


# === ROUTES ===


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        # Load Whisper with Indonesian language support (large / turbo)
        # https://github.com/openai/whisper
        # whisper_model = whisper.load_model("large")

        # Start measuring time
        start_time = time.time()

        audio_file = request.files["file"]
        if audio_file:
            # Save uploaded audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_file.read())
                temp_audio_path = temp_audio.name

            # Step 1: Transcribe
            transcribed_text = faster_whisper(temp_audio_path)
            os.remove(temp_audio_path)

            # Step 2: BERT Prediction
            predicted_label, prob_results = bert_predict(transcribed_text)

            # Stop timer
            end_time = time.time()
            total_time = end_time - start_time
            formatted_time = f"{total_time:.2f} seconds"

            return render_template(
                "transcribe.html",
                task=transcribed_text,
                prediction=predicted_label,
                probabilities=prob_results,
                total_time=formatted_time,
            )

    except Exception as e:
        print("Error:", e)
        return str(e)


@app.route("/predict-text", methods=["POST"])
def predict_text():
    try:
        user_lyrics = request.form.get("lyrics", "").strip()

        if not user_lyrics:
            return "No lyrics provided.", 400

        # Start timer
        start_time = time.time()

        # Step 1: BERT Prediction
        predicted_label, prob_results = bert_predict(user_lyrics)

        # End timer
        end_time = time.time()
        total_time = f"{end_time - start_time:.2f} seconds"

        return render_template(
            "transcribe.html",
            task=user_lyrics,
            prediction=predicted_label,
            probabilities=prob_results,
            total_time=total_time,
        )

    except Exception as e:
        print("❌ Error in predict-text:", e)
        return str(e), 500


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm-password")

        if User.query.filter_by(email=email).first():
            return render_template(
                "register.html",
                error="Email already taken!",
                email=email,
                password=password,
                confirm_password=confirm_password,
            )

        if password != confirm_password:
            return render_template(
                "register.html",
                error="Password does not match!",
                email=email,
                password=password,
                confirm_password=confirm_password,
            )

        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return dashboard(login_alert=True)
        else:
            return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")


def dashboard(login_alert=False):
    if login_alert:
        print('test')
        return render_template("index.html", email=current_user.email)
    return redirect(url_for("index"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/history", methods=["GET"])
def history():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
