from transformers import pipeline
import torch

class AnswerGenerator:
    def __init__(self):
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if torch.cuda.is_available() else -1
        )

    def generate_answer(self, question):
        prompt = f"Create five age-appropriate answer choices for the question: '{question}' The options should reflect different emotions that children might experience."
        generated = self.generator(prompt, max_length=50, num_beams=5, num_return_sequences=1)
        return generated[0]['generated_text']
