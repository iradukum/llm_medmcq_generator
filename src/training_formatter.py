import json
import os
import ast
import pandas as pd
from typing import Dict, List, Union


class TrainingFormatter:
    """
    Converts MCQ records into {prompt, completion} format
    for supervised fine-tuning of Phi2 LLM.
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

    def format_mcq(self, mcq: Dict[str, str]) -> Dict[str, str]:
        """
        Format a single mcq into {prompt, completion} dict
        """
        prompt = (
            f"### Summary:\n{mcq["summary"]}\n"
            f"### Learning Objective:\n{mcq["objective"]}\n"
            f"### Instruction:\n{self.BASE_INSTRUCTION}"
        )
        options = ast.literal_eval(mcq["options"])
        formatted_options = "\n".join([f"{k}. {v}" for k, v in options.items()])
        completion = (
            f"Question: {mcq["question"]}\n"
            f"Options:\n{formatted_options}\n"
            f"Answer: {mcq["answer"]}\n"
            f"Explanation: {mcq["explanation"]}"
        )
        return {"prompt": prompt.strip(), "completion": completion.strip()}


    def format_dataframe(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Format an entire dataframe of MCQs.
        """
        return [self.format_mcq(mcq) for _, mcq in df.iterrows()]


    def save_jsonl(self, formatted_mcqs: List[Dict[str, str]], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for mcq in formatted_mcqs:
                f.write(json.dumps(mcq) + '\n')
        print(f"Saved {len(formatted_mcqs)} formatted mcq in {path}")
        