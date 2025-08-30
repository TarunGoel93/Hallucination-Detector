from flask import Flask, render_template, request
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    pipeline = joblib.load("hallucination_detector_char_ngram.pkl")
    print("Loaded Logistic Regression model from hallucination_detector_char_ngram.pkl")
except FileNotFoundError:
    print("ERROR: Model file 'hallucination_detector_char_ngram.pkl' not found.")
    exit()

# Function to predict hallucination
def predict_hallucination(src, hyp, tgt="", threshold=0.5):
    concat_text = f"Source: {src}\nHypothesis: {hyp}\nTarget: {tgt if tgt else ''}"
    prob = pipeline.predict_proba([concat_text])[0][1]  # Probability of hallucination
    label = "Hallucination" if prob >= threshold else "Not Hallucination"
    return label, prob

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = 0.0
    confidence_color = "gray"  # Default color

    if request.method == "POST":
        src = request.form.get("src", "")
        hyp = request.form.get("hyp", "")
        tgt = request.form.get("tgt", "")  # Optional target
        prediction, probability = predict_hallucination(src, hyp, tgt)
        # Dynamic confidence color: green (>0.9), yellow (0.6-0.9), red (<0.6)
        if probability > 0.9:
            confidence_color = "green"
        elif probability > 0.6:
            confidence_color = "yellow"
        else:
            confidence_color = "red"

    return render_template("index.html", prediction=prediction, probability=probability, confidence_color=confidence_color)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)