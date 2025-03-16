from flask import Flask, render_template, url_for, request, redirect
# from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

import whisper
import tempfile
import os

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
# db = SQLAlchemy(app)

# class Todo(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     content = db.Column(db.String(200), nullable=False)
#     date_created = db.Column(db.DateTime, default=datetime.utcnow)

#     def __repr__(self):
#         return '<Task %r>' % self.id


@app.route('/', methods=['POST', 'GET'])
def index():
    # if request.method == 'POST':
    #     task_content = request.form['content']
    #     new_task = Todo(content=task_content)

    #     try:
    #         db.session.add(new_task)
    #         db.session.commit()
    #         return redirect('/')
    #     except:
    #         return 'There was an issue adding your task'

    # else:
    #     tasks = Todo.query.order_by(Todo.date_created).all()
    #     return render_template('index.html', tasks=tasks)
    return render_template('index.html')


# @app.route('/delete/<int:id>')
# def delete(id):
#     task_to_delete = Todo.query.get_or_404(id)

#     try:
#         db.session.delete(task_to_delete)
#         db.session.commit()
#         return redirect('/')
#     except:
#         return 'There was a problem deleting that task'

# @app.route('/update/<int:id>', methods=['GET', 'POST'])
# def update(id):
#     task = Todo.query.get_or_404(id)

#     if request.method == 'POST':
#         task.content = request.form['content']

#         try:
#             db.session.commit()
#             return redirect('/')
#         except:
#             return 'There was an issue updating your task'

#     else:
#         return render_template('update.html', task=task)

@app.route('/transcribe', methods=['GET', 'POST'])
def transcribe():
    if request.method == 'POST':
        try:
            model = whisper.load_model("turbo")
            audio_file = request.files['file']
            if audio_file is not None:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio.write(audio_file.read())
                    temp_audio_path = temp_audio.name

                transcription = model.transcribe(temp_audio_path, language="id")
                # Clean up temp file
                os.remove(temp_audio_path)
            return render_template('transcribe.html', task=transcription["text"])
        except Exception as error:
            print("An error occurred:", error)
            return 'There was an issue updating your task' + error

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
