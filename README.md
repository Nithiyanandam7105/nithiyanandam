import pandas as pd
import nltk
import torch
from transformers import pipeline
from nltk.tokenize import sent_tokenize

# Download NLTK sentence tokenizer
nltk.download('punkt')

# Example social media text (replace with real data)
texts = [
    "I'm so excited about the concert tonight! Can't wait!",
    "Ugh, my flight got canceled again. Why is this happening?",
    "I feel really down today... nothing seems to help.",
    "Thank you all for your support. I feel truly blessed.",
    "Why is nobody talking about the climate crisis?"
]

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Load emotion classification model from HuggingFace
emotion_pipeline = pipeline("text-classification", 
                            model="j-hartmann/emotion-english-distilroberta-base", 
                            return_all_scores=True)

# Analyze each text
results = []

for text in texts:
    sentiments = sentiment_pipeline(text)
    emotions = emotion_pipeline(text)

    top_emotion = max(emotions[0], key=lambda x: x['score'])
    top_sentiment = sentiments[0]

    results.append({
        "text": text,
        "sentiment": top_sentiment['label'],
        "sentiment_score": round(top_sentiment['score'], 3),
        "emotion": top_emotion['label'],
        "emotion_score": round(top_emotion['score'], 3)
    })

# Convert to DataFrame
df = pd.DataFrame(results)

print(df)
