# backend/test_analyzer.py

from models.analyzer import Analyzer

def test_analyzer():
    analyzer = Analyzer()
    responses = {
        "How was your day at school today?": "A- It was great!",
        "Did you make any new friends?": "B- It was okay.",
        "Did you enjoy your classes?": "C- It was bad.",
        # Add more responses as needed
    }
    analysis = analyzer.analyze_responses(responses)
    print(f"Analysis: {analysis}")

if __name__ == "__main__":
    test_analyzer()
