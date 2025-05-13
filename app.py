from flask import Flask, render_template, request
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load Hugging Face emotion detection model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

@app.route("/", methods=["GET", "POST"])
def index():
    emotion_result = None
    user_post = ""

    if request.method == "POST":
        user_post = request.form.get("post", "")
        if user_post.strip():
            try:
                result = emotion_classifier(user_post)
                emotion_result = result[0]['label']
            except Exception as e:
                emotion_result = "Error detecting emotion"

    return render_template("index.html", emotion=emotion_result, post=user_post)

# Run the app locally (this line is ignored by Render which uses gunicorn)
if __name__ == "__main__":
    app.run(debug=True)
