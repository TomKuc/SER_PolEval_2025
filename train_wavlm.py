import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    WavLMForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from datasets import concatenate_datasets

# ===== PATHS =====
BASE_PATH = "PATH_TO_SER_DATA"
TRAIN_TSV = os.path.join(BASE_PATH, "train", "in.tsv")
DEV_TSV   = os.path.join(BASE_PATH, "dev", "in.tsv")

OUTPUT_DIR = "./outputs/wavlm_finetune"
MODEL_NAME = "microsoft/wavlm-base-plus"
NUM_LABELS = 6
LABELS = ["anger", "fear", "happiness", "sadness", "surprise", "neutral"]

# ===== LOADING FILE LISTS =====
train_ids = pd.read_csv(TRAIN_TSV, sep="\t", header=None, names=["split", "file_id"])
dev_ids   = pd.read_csv(DEV_TSV, sep="\t", header=None, names=["split", "file_id"])

# ===== LOADING DATA FROM HUGGINGFACE =====
def load_split_data(ids_df):
    """Loads data from the corresponding CAMEO splits"""
    data_list = []
    for split_name in ids_df["split"].unique():
        ds = load_dataset("amu-cai/CAMEO", split=split_name)
        ids_for_split = set(ids_df[ids_df["split"] == split_name]["file_id"])
        ds = ds.filter(lambda ex: ex["file_id"] in ids_for_split and ex["emotion"] in LABELS)
        data_list.append(ds)
    return concatenate_datasets(data_list)

print("Loading train...")
train_ds = load_split_data(train_ids)

print("Loading dev (Polish)...")
dev_ds = load_split_data(dev_ids)

# ===== PROCESSOR =====
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

def preprocess(batch):
    audio = batch["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, max_length=int(3.5*16000),
                       truncation=True, padding="max_length")
    batch["input_values"] = inputs["input_values"][0]
    batch["attention_mask"] = inputs["attention_mask"][0]
    batch["labels"] = LABELS.index(batch["emotion"])
    return batch

print("Processing data...")
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
dev_ds = dev_ds.map(preprocess, remove_columns=dev_ds.column_names)

train_ds.set_format(type="torch")
dev_ds.set_format(type="torch")

# ===== MODEL =====
print("Initializing model...")
model = WavLMForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

# ===== METRICS =====
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

# ===== TRAINING ARGUMENTS =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=2
)

# ===== TRAINER =====
print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    tokenizer=processor,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(f"{OUTPUT_DIR}/final_model")
print("Saved best model to:", f"{OUTPUT_DIR}/final_model")
