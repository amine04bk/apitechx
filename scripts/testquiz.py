import os
import logging
from transformers import pipeline
import torch


# Suppress TensorFlow and Transformers logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
logging.getLogger("transformers").setLevel(logging.ERROR)  # Suppress Transformers logs

# Initialize Hugging Face pipelines with specific free models
question_generator = pipeline(
    "text2text-generation",
    model="t5-small",
    device=0 if torch.cuda.is_available() else -1  # Utilize GPU if available
)

answer_generator = pipeline(
    "text2text-generation",
    model="t5-small",
    device=0 if torch.cuda.is_available() else -1  # Utilize GPU if available
)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1  # Utilize GPU if available
)

# Generate the next question based on the previous question and answer
def generate_next_question(previous_question, previous_answer):
    prompt = (
        f"Given the previous question and the child's answer, generate a simple and clear follow-up question for the child.\n\n"
        f"Previous Question: {previous_question}\n"
        f"Child's Answer: {previous_answer}\n"
        f"Next Question:"
    )
    generated_text = question_generator(
        prompt,
        max_length=50,
        num_return_sequences=1,
        truncation=True,
        clean_up_tokenization_spaces=True
    )
    next_question = generated_text[0]['generated_text'].strip()
    return next_question

# Generate multiple-choice answers for each question
def generate_answers(question):
    prompt = (
        f"Create 5 simple and clear multiple-choice answers (A-E) for the following question.\n\n"
        f"Question: {question}\n"
        f"Answers:"
    )
    try:
        generated_text = answer_generator(
            prompt,
            max_length=150,
            num_return_sequences=1,
            truncation=True,
            clean_up_tokenization_spaces=True
        )

        answers_text = generated_text[0]['generated_text'].strip()

        # Split the answers by lines or common separators
        answers = []
        for line in answers_text.split('\n'):
            line = line.strip()
            if line:
                # Ensure each answer starts with A-, B-, etc.
                if line[0] in ['A', 'B', 'C', 'D', 'E'] and line[1] in ['-', ')']:
                    answers.append(line)
                else:
                    # If not labeled, add labels
                    answers.append(f"{chr(65 + len(answers))}- {line}")
            if len(answers) == 5:
                break

        # If fewer than 5 answers are generated, pad with "I don't know"
        while len(answers) < 5:
            answers.append(f"{chr(65 + len(answers))}- I don't know")

        return answers

    except Exception as e:
        print(f"Error generating answers: {e}")
        return [
            "A- I don't know",
            "B- I don't know",
            "C- I don't know",
            "D- I don't know",
            "E- I don't know"
        ]

# Analyze the child's responses to determine mental state
def analyze_responses(responses):
    combined_responses = " ".join([f"Question: {q}, Answer: {a}" for q, a in responses.items()])
    analysis = sentiment_analyzer(combined_responses)
    positive_count = sum(1 for result in analysis if result['label'] in ['4 stars', '5 stars'])
    negative_count = sum(1 for result in analysis if result['label'] in ['1 star', '2 stars'])
    advice = ""

    if positive_count > negative_count:
        advice = "Great job! Your child had a positive interaction. Encourage their engagement and support them further."
    elif negative_count > positive_count:
        advice = "It seems your child may be experiencing some challenges. Consider discussing their feelings to provide support."
    else:
        advice = "Your child's responses seem balanced. Keep communicating to understand their thoughts better."

    return advice

def main():
    initial_question = "How do you feel today?"
    responses = {}

    current_question = initial_question

    for i in range(10):  # Limit to 10 questions for the session
        print(f"\nQuestion {i+1}: {current_question}")
        answers = generate_answers(current_question)

        # Display clean and relevant answers
        for answer in answers:
            print(answer)

        # Input validation to ensure user picks a valid option
        while True:
            choice = input("I choose... ").strip().upper()
            if choice in [chr(65 + i) for i in range(len(answers))]:  # Validating if the choice is A-E
                break
            else:
                print("Please select a valid option (A-E).")

        # Store the current question and answer
        selected_answer = next(
            (ans for ans in answers if ans.startswith(choice)),
            "I don't know"
        )
        responses[current_question] = selected_answer

        # Generate the next question based on the current one and the chosen answer
        current_question = generate_next_question(current_question, selected_answer)

    # Analyze the responses
    analysis = analyze_responses(responses)
    print("\nAnalysis of the child's mental state:")
    print(analysis)

if __name__ == "__main__":
    main()
