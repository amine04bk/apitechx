# backend/models/analyzer.py

from transformers import pipeline
import torch

class Analyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1
        )

    def analyze_responses(self, responses):
        combined_responses = " ".join([f"Question: {q}, Answer: {a}" for q, a in responses.items()])
        analysis = self.sentiment_analyzer(combined_responses)
        positive_count = sum(1 for result in analysis if result['label'] in ['4 stars', '5 stars'])
        negative_count = sum(1 for result in analysis if result['label'] in ['1 star', '2 stars'])
        if positive_count > negative_count:
            advice = "Great job! Your child had a positive interaction. Encourage their engagement and support them further."
        elif negative_count > positive_count:
            advice = "It seems your child may be experiencing some challenges. Consider discussing their feelings to provide support."
        else:
            advice = "Your child's responses seem balanced. Keep communicating to understand their thoughts better."
        return advice
