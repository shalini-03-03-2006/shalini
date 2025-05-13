from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

@app.route("/", methods=["GET", "POST"])
def index():
    emotion_result = None
    if request.method == "POST":
        user_post = request.form.get("post")
        if user_post:
            result = emotion_classifier(user_post)
            emotion_result = result[0]['label']
    return render_template("index.html", emotion=emotion_result)
