import os
import ollama
from typing import List, Tuple, Optional


class SummaryGenerator:
    """
    Generate medical summaries from a group of QA pairs using Llama3.
    """

    BASE_HEADER = """
    You are a medical assistant helping to create educational summaries.
    Given a set of medical question-answer pairs, generate a short, clinically realistic summary (100–150 words)
    that captures the core topic and useful diagnostic or therapeutic information.
    """

    BASE_INSTRUCTION = """
    Use the Q&A pairs below to synthesize.
    Write in neutral tone, no bullet points, no listing. Avoid repeating questions.
    
    """

    def __init__(self, model: str = "llama3.1"):
        self.model = model

    def build_prompt(self, qa_pairs: List[Tuple[str, str]]) -> str:
        prompt = self.BASE_HEADER + self.BASE_INSTRUCTION
        for i, (q, a) in enumerate(qa_pairs):
            prompt += f"Q{i+1}: {q}\nA{i+1}: {a}\n"
        prompt += "\nSummary:\n"
        return prompt

    def generate_summary(self, qa_pairs: List[Tuple[str, str]]) -> Optional[str]:
        if not qa_pairs:
            return None

        selected_qas = qa_pairs[:10]  # Limit token size for llama input
        prompt = self.build_prompt(selected_qas)
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[ERROR] Summary generation failed: {e}")
            return None