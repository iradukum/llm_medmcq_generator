import re
import pandas as pd


class QADataPreprocessor:
    """
    Preprocesses question-answer pairs by cleaning text and punctuation.
    Useful for preparing QA data before MCQ generation or model training.
    """
    
    def __init__(self, text_col_question: str = "question", text_col_answer: str = "answer"):
        """
        Args:
            text_col_question (str): Column name for the question text.
            text_col_answer (str): Column name for the answer text.
        """
        self.q_col = text_col_question
        self.a_col = text_col_answer


    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())       # Remove any sequence of whitespace 
        text = re.sub(r"\n+", " ", text)               # Remove one or more newlines
        text = re.sub(r" +", " ", text)                # Remove multiple spaces
        return text


    def clean_punctuation(self, text: str) -> str:
        text = re.sub(r"\s*([.,!?;:])\s*", r"\1 ", text)  # Tighten spacing around punctuation, e.g. "word , word" -> "word, word"
        text = re.sub(r"([?!.,]){2,}", r"\1", text)        # Collapse repeated punctuation, e.g. "Really????" -> "Really?"
        text = re.sub(r"['\"]", "", text)                 # Remove quotes
        return text.strip()


    def process_row(self, row: pd.Series) -> pd.Series:
        row[self.q_col] = self.clean_punctuation(self.clean_text(row[self.q_col]))
        row[self.a_col] = self.clean_punctuation(self.clean_text(row[self.a_col]))
        return row


    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.apply(self.process_row, axis=1)
        df.drop_duplicates(subset=[self.q_col, self.a_col], inplace=True)
        return df
        

    def save_clean_csv(self, df:pd.DataFrame, path: str):
        df.to_csv(path, index=False)
