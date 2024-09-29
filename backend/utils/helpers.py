# backend/utils/helpers.py

def generate_initial_questions(qg, ag, num_questions=10):
    questions = []
    answers = []
    initial_prompt = "Generate a simple, child-friendly question to assess a child's feelings or thoughts."

    for _ in range(num_questions):
        question = qg.generate_question(initial_prompt)
        answer_options = ag.generate_answers(question)
        questions.append(question)
        answers.append(answer_options)
    return questions, answers
