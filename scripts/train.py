import os
import json
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


class Phi2Trainer:
    """
    Fine-tunes the Phi-2 model using LoRA (QLoRA-ready).
    """
    def __init__(self, model_name: str, train_file: str, eval_file: str, output_dir: str):
        """
        Args:
            model_name (str): model name or local checkpoint (e.g. 'microsoft/phi-2')
            train_file (str): Path to training JSONL file (prompt/completion)
            eval_file (str): Path to evaluation JSONL file
            output_dir (str): Path to store model checkpoints
        """
        self.train_df = pd.read_json(train_file, lines=True)
        self.eval_df = pd.read_json(eval_file, lines=True)
        self.output_dir = output_dir
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def _tokenize(self, example):
        return self.tokenizer(
            f"{example['prompt']}\n{example['completion']}",
            padding="max_length",
            truncation=True,
            max_length=512,
        )


    def train(self):
        # Prepare tokenized datasets
        train_dataset = Dataset.from_pandas(self.train_df)
        train_dataset = train_dataset.map(self._tokenize, remove_columns=train_dataset.column_names)
        eval_dataset = Dataset.from_pandas(self.eval_df)
        eval_dataset = eval_dataset.map(self._tokenize, remove_columns=eval_dataset.column_names)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map="auto",
        )

         # Apply LoRA for parameter-efficient tuning
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=10,
            save_steps=100,
            save_total_limit=1,
            logging_dir=os.path.join(self.output_dir, "logs"),
            report_to="tensorboard",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        )

        trainer.train()

        # Save final model and tokenizer
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print("Training complete. Model and tokenizer saved.")


if __name__ == "__main__":
    MODEL_NAME = "microsoft/phi-2"
    TRAIN_FILE = "data/training_jsonl/mcqs_filtered_07.jsonl"
    EVAL_FILE = "data/training_jsonl/mcqs_filtered_085.jsonl"
    OUTPUT_DIR = "models/phi2_mcq_sft"

    phi2_trainer = Phi2Trainer(MODEL_NAME, TRAIN_FILE, EVAL_FILE, OUTPUT_DIR)
    phi2_trainer.train()
            