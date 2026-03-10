import gradio as gr
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model
calibrated_spam = joblib.load("Model/spam_classifier_calibrated.joblib")

# Preprocessing + feature builder
def build_features(message: str):
    cleaned = message.lower()
    # Remove non-alphabetic characters except spaces
    cleaned = ''.join(c for c in cleaned if c.isalpha() or c.isspace())
    
    return {
        'clean_text': cleaned,
        'length': len(cleaned),
        'word_count': len(cleaned.split()),
        'has_free': int('free' in cleaned),
        'has_win': int(any(w in cleaned for w in ['win', 'won', 'prize'])),
        'exclamation': message.count('!')
    }

# Prediction function
def predict_spam_single(message: str):
    if not message.strip():
        return "Please enter a message", "", 0.0
    
    features = pd.DataFrame([build_features(message)])
    
    proba = calibrated_spam.predict_proba(features)[0][1]
    label = "Spam" if proba > 0.5 else "Ham"
    confidence_pct = round(proba * 100, 1)
    
    result_text = f"**{label}** (confidence: {confidence_pct}%)"
    explanation = (
        "This message shows typical spam patterns (keywords like free/win/prize, excessive punctuation, etc.)."
        if label == "Spam"
        else "This appears to be a legitimate (ham) message."
    )
    
    return result_text, explanation, confidence_pct

# Gradio UI
with gr.Blocks(title="Spam Email Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # SMS / Email Spam Detector

        Enter a message below to check if it's **Spam** or **Ham** (legitimate).

        Built with scikit-learn • Lightweight & fast • Calibrated probabilities
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="Message",
                placeholder="Type or paste a message here...",
                lines=4,
                max_lines=8,
                autofocus=True
            )
            btn = gr.Button("Classify", variant="primary", scale=0)
        
        with gr.Column(scale=2):
            output_label = gr.Markdown(label="Prediction")
            output_explanation = gr.Markdown(label="Why this prediction?")
            confidence_slider = gr.Slider(
                0, 100, value=0,
                label="Confidence (%)",
                interactive=False
            )
    
    gr.Examples(
        examples=[
            ["WINNER!! You are selected for a free £1000 prize. Claim now!"],
            ["Hey bro, are we still meeting at 7 pm tonight?"],
            ["Congratulations! Your mobile number has won £2000 prize. Text YES to 12345"],
            ["Just checking in, how's everything going?"],
            ["URGENT! Your account has been suspended. Verify now: bit.ly/abc123"],
            ["Meeting at 3 PM in conference room B"]
        ],
        inputs=input_text,
        label="Try these examples"
    )
    
    btn.click(
        fn=predict_spam_single,
        inputs=input_text,
        outputs=[output_label, output_explanation, confidence_slider]
    )

demo.launch()
