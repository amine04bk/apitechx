# backend/test_question_generator.py

from models.question_generator import QuestionGenerator

def test_question_generation():
    qg = QuestionGenerator()
    prompt = "Generate a simple and open-ended question for a child that encourages them to express their feelings and thoughts about their day. For example, ask how they felt during a particular activity or event. the question generated for answering by a kid"
    question = qg.generate_question(prompt)
    print(f"Generated Question: {question}")

if __name__ == "__main__":
    test_question_generation()
