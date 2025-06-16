import re
import pandas as pd


class QADataPreprocessor:
    def __init__(self, text_col_question: str = "question", text_col_answer: str = "answer"):
        self.q_col = text_col_question
        self.a_col = text_col_answer

    def clean_text(self, text: str) -> str:
        # Normalize unicode, collapse spaces, strip
        text = re.sub(r"\s+", " ", text.strip()) # Replaces any sequence of whitespace with a single space
        text = re.sub(r"\n+", " ", text) # Replaces one or more newlines with a single space
        text = re.sub(r" +", " ", text) # Replaces multiple spaces with just one
        return text

    def clean_punctuation(self, text: str) -> str:
        # Remove excessive punctuaction spacing, strip trailing dots, fix quotes
        text = re.sub(r"\s*([.,!?;:])\s*", r"\1 ", text) # reuses the punctuation match (inside parentheses), then adds 1 space after, e.g. "word , word" -> "word, word"
        text = re.sub(r"([?!.,]){2,}", r"\1", text) # Matches repeated punctuation (like !! or ??) and replaces with one, e.g. "Really????" -> "Really?"
        text = re.sub(r"['\"]", "", text) # Removes both single and double quotes
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
