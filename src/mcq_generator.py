import ollama
from typing import List, Tuple, Dict, Optional



class MCQGenerator:
    """
    Generate medical MCQs using a local LLaMA 3.1 model via Ollama.
    Learning objective is derived from Q-types.
    Ensures plausible distractors and higher-order clinical reasoning.
    """

    BASE_INSTRUCTION = """
        You are a senior medical question writer for board-style exams.
        Given a clinical summary and a set of question types, generate a multiple-choice question with the following requirements:
        
        - High cognitive level (application, analysis, synthesis)
        - Medically accurate and educationally relevant
        - One clearly correct answer
        - Three distractors that are plausible but clearly incorrect
        - Label options A–D
        - Include a concise explanation justifying the correct answer
        
        Here is the mandatory output format expected all in neither bold nor italic writing, just normal writing:
        Objective: <...>
        Question: <...>
        Options:
        A. ...
        B. ...
        C. ...
        D. ...
        Answer: <A/B/C/D>
        Explanation: <...>
    """

    def __init__(self, model: str = "llama3.1"):
        self.model = model

    
    def build_prompt(self, summary: str, qtypes: List[str]) -> Tuple[str, str]:
        prompt = f"{self.BASE_INSTRUCTION}\n\nClinical Summary:\n{summary}\n\nRelevant Question Types: {', '.join(qtypes)}\n"
        return prompt

    def generate_mcq(self, summary: str, qtypes: List[str]) -> Optional[Dict[str, str]]:
        prompt = self.build_prompt(summary, qtypes)

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
            return self.parse_output(response["message"]["content"])
        except Exception as e:
            print(f"[ERROR] MCQ generation failed: {e}")
            return None

    def parse_output(self, text: str) -> Dict[str, str]:
        lines = text.strip().splitlines()
        mcq = {"options": {}}
        for line in lines:
            if line.startswith("Objective:"):
                mcq["objective"] = line.replace("Objective:", "").strip()
            elif line.startswith("Question:"):
                mcq["question"] = line.replace("Question:", "").strip()
            elif line[:2] in ["A.", "B.", "C.", "D."]:
                mcq["options"][line[0]] = line[3:].strip()
            elif line.startswith("Answer:"):
                mcq["answer"] = line.replace("Answer:", "").strip()
            elif line.startswith("Explanation:"):
                mcq["explanation"] = line.replace("Explanation:", "").strip()
        return mcq

    
    

    