import ollama
from typing import List, Tuple, Dict, Optional



class MCQGenerator:
    """
    Generate medical MCQs using a local LLaMA 3.1 model via Ollama, by respecting these requirements:
    - Objective derived from question types
    - Medically accurate, high-cognitive-level MCQs
    - One correct option (A–D) with explanation
    """

    BASE_INSTRUCTION = """
        You are a senior medical question writer for board-style exams.
        Given a clinical summary and a set of question types, generate a multiple-choice question with the following requirements:

        - The objective has to be derived from the question types
        - The multiple choice-question has to answer to the chosen objective and based on the summary given
        - High cognitive level (application, analysis, synthesis)
        - Medically accurate and educationally relevant
        - One clearly correct answer
        - Three distractors that are plausible but clearly incorrect
        - Label options A–D
        - Include a concise explanation justifying the correct answer
        
        Here is the mandatory output format expected all in neither bold nor italic writing, just normal writing. On the first line we should have the objective, the question on the following line, the options on the following lines, the answer on another line and finally the explanation on the last line. Here is the format expected:
        Objective: ...
        Question: ...
        Options:
        A. ...
        B. ...
        C. ...
        D. ...
        Answer: <A/B/C/D>
        Explanation: ...
    """

    def __init__(self, model: str = "llama3.1"):
        self.model = model

    
    def build_prompt(self, summary: str, qtypes: List[str]) -> Tuple[str, str]:
        prompt = f"{self.BASE_INSTRUCTION}\n\nClinical Summary:\n{summary}\n\nRelevant Question Types: {', '.join(qtypes)}\n"
        return prompt

    def generate_mcq(self, summary: str, qtypes: List[str], max_retries: int = 2) -> Optional[Dict[str, str]]:
        prompt = self.build_prompt(summary, qtypes)
        for attempt in range(max_retries + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                mcq = self.parse_output(response["message"]["content"])

                if self._is_valid(mcq):
                    return mcq
                elif attempt < max_retries:
                    print(f"[WARNING] Incomplete MCQ on attempt {attempt+1} → Retrying...")
            except Exception as e:
                print(f"[ERROR] Generation failed (attempt {attempt+1}): {e}")
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

    def _is_valid(self, mcq: Dict[str, str]) -> bool:
        required_keys = ["objective", "question", "options", "answer", "explanation"]
        return all(k in mcq and mcq[k] for k in required_keys if k != "options") and len(mcq.get("options", {})) == 4

    
    

    