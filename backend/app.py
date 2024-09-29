# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from models.question_generator import QuestionGenerator
from models.answer_generator import AnswerGenerator
from models.analyzer import Analyzer
from utils.helpers import generate_initial_questions
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AI model instances
logger.info("Initializing AI models...")
question_generator = QuestionGenerator()
answer_generator = AnswerGenerator()
analyzer = Analyzer()
logger.info("AI models initialized successfully.")

@app.route('/generate_quiz', methods=['GET'])
def generate_quiz():
    try:
        num_questions = 10
        logger.info(f"Generating {num_questions} quiz questions.")
        questions, answer_options = generate_initial_questions(question_generator, answer_generator, num_questions)
        quiz = []
        for i in range(num_questions):
            quiz.append({
                'question_number': i + 1,
                'question': questions[i],
                'answers': answer_options[i]
            })
        logger.info("Quiz generated successfully.")
        return jsonify({'status': 'success', 'quiz': quiz}), 200
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/analyze_responses', methods=['POST'])
def analyze_responses():
    try:
        data = request.get_json()
        responses = data.get('responses')  # Expected format: { "Question 1": "A- Answer", ... }
        if not responses:
            logger.warning("No responses provided in the request.")
            return jsonify({'status': 'error', 'message': 'No responses provided.'}), 400
        logger.info("Analyzing responses.")
        analysis = analyzer.analyze_responses(responses)
        logger.info("Analysis completed.")
        return jsonify({'status': 'success', 'analysis': analysis}), 200
    except Exception as e:
        logger.error(f"Error analyzing responses: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
c. Testing the AI Models Independently
Before integrating the models into the backend, it's crucial to test them independently to ensure they generate the desired output.

i. Testing Question Generation
python
Copier le code
# backend/test_question_generator.py

from models.question_generator import QuestionGenerator

def test_question_generation():
    qg = QuestionGenerator()
    prompt = "Generate a simple, child-friendly question to assess a child's feelings or thoughts."
    question = qg.generate_question(prompt)
    print(f"Generated Question: {question}")
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
