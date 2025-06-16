import os
import pandas as pd
from bs4 import BeautifulSoup


def _extract_qa_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml-xml")
        doc = soup.find("Document")
        
        if not doc:
            return []

        doc_id = doc.get("id")
        source = doc.get("source")
        focus = doc.find("Focus").text.strip() if doc.find("Focus") else None

        qa_pairs = []

        for qa in doc.find_all("QAPair"):
            q_id = qa.find("Question").get("qid", "")
            q_type = qa.find("Question").get("qtype", "")
            question = qa.find("Question").text.strip() if qa.find("Question") else None
            answer = qa.find("Answer").text.strip() if qa.find("Answer") else None

            if question and answer:
                qa_pairs.append({
                    "question_id": q_id,
                    "document_id" : doc_id,
                    "source" : source,
                    "focus" : focus,
                    "question_type": q_type,
                    "question": question,
                    "answer": answer
                })
        return qa_pairs


def build_dataset(root_dir: str="../data/MedQuAD-master") -> pd.DataFrame:
    
    all_data = []
    
    for subfolder in os.listdir(root_dir):
        
        folder_path = os.path.join(root_dir, subfolder)
        
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".xml"):
                    file_path = os.path.join(folder_path, file)
                    all_data.extend(_extract_qa_from_file(file_path))

    return pd.DataFrame(all_data)


def save_to_csv(df: pd.DataFrame, out_path: str="../data/parsed_csv/qa_flat.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


