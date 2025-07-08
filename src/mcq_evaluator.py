import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from typing import Dict, List
from sentence_transformers import SentenceTransformer


nltk.download("punkt_tab", quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)


class MCQEvaluator:
    """
    Evaluate the quality of medical multiple-choice questions using semantic similarity,
    lexical checks, and domain-specific criteria.

    Scores:
        - relevance_to_summary: how semantically aligned the question is with the original summary
        - alignement_with_objective: how well the question satisfies the educational objective
        - plausibility_of_distractors: how realistic yet incorrect distractors are
        - plausibility_qa: basic sanity checks for question structure and answer
        - medical_validity: whether explanation includes medical terminology
    """

    def __init__(self):
        self.sent_transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


    def evaluate(self, mcq: Dict[str, str], summary: str, objective: str) -> Dict[str, float]:
        """
        Args:
            mcq (Dict[str, str]): Parsed MCQ dictionary with fields like 'question', 'options', etc.
            summary (str): Source summary text the MCQ should be based on.
            objective (str): Instructional learning objective.

        Returns:
            Dict[str, float]: Component scores and averaged score.
        """
        scores = {}
        scores["relevance_to_summary"] = self._semantic_score(summary, mcq["question"])
        scores["alignement_with_objective"] = self._semantic_score(objective, mcq["question"])
        scores["plausibility_of_distractors"] = self._distractor_score(mcq["question"], mcq["options"], mcq["answer"][0])
        scores["plausibility_qa"] = self._qa_score(mcq["question"], mcq["options"], mcq["answer"])
        scores["medical_validity"] = self._validity_score(mcq["explanation"])
        scores["average"] = round(np.mean(list(scores.values())), 3)
        return scores


    def _semantic_score(self, ref: str, test: str) -> float:
        emb1 = self.sent_transformer_model.encode(ref, convert_to_tensor=True)
        emb2 = self.sent_transformer_model.encode(test, convert_to_tensor=True)
        return float(self.sent_transformer_model.similarity(emb1, emb2)[0][0])


    def _distractor_score(self, question: str, options: Dict[str, str], answer: str) -> float:
        distractors = [v for k, v in options.items() if k != answer]
        answer_text = options.get(answer)
        scores = []
        for d in distractors:
            q_sim = self._semantic_score(d, question)
            a_sim = self._semantic_score(d, answer_text)
            if q_sim > 0.3 and a_sim < 0.7:
                scores.append(1.0)
            elif q_sim > 0.2 and a_sim < 0.8:
                scores.append(0.75)
            else:
                scores.append(0.5)
        return round(float(np.mean(scores)), 3)


    def _qa_score(self, question: str, options: Dict[str, str], answer: str) -> float:
        if answer not in options:
            return 0.0
        questions_tokens = word_tokenize(question)
        if len(questions_tokens) < 5:
            return 0.5
        return 1.0

    def _get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:         
            return wordnet.NOUN

    def _validity_score(self, explanation: str) -> float:
        medical_keywords = set([
            "diagnosis", "diagnose", "symptom", "treatment", "pathology", "risk",
            "screening", "prognosis", "medication", "disease",
            "anatomy", "biopsy", "therapeutic", "chronic", "acute",
            "infection", "lesion", "neoplasm", "inflammation", "vaccine",
            "syndrome", "patient", "gene", "genetic", "mutation", "mutate",
            "mutated", "clinical", "disorder", "therapy", "recessive", "complication",
            "result", "cell", "surgery", "surgical", "tumor", "cancer", "pain",
            "congenital", "intervention"
        ])
        
        lemmatizer = WordNetLemmatizer()
        pos_tags = pos_tag(word_tokenize(explanation.lower()))
        lemmatized_tokens = set([lemmatizer.lemmatize(word, self._get_wordnet_pos(tag)) for word, tag in pos_tags])
        intersection = medical_keywords.intersection(lemmatized_tokens)
        return 1.0 if len(intersection) >= 3 else 0.5

    
    def parse_to_mcq(self, text: str) -> Dict[str, str]:
        """
        Parse model output string into structured MCQ dictionary.

        Returns:
            Dict[str, str]: A dict with keys: objective, question, options, answer, explanation
        """
        lines = text.strip().splitlines()
        mcq = {"options": {}}
        for line in lines:
            line = line.strip()
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
    