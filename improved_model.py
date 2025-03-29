import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
import nltk
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import random
import os

# Download NLTK resources
nltk.download('punkt')

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Create output directory
os.makedirs("improved-bart-lora", exist_ok=True)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("csv", data_files="extracted_data.csv")

# Rename columns for compatibility
dataset = dataset.rename_column("XML_Text", "document")
dataset = dataset.rename_column("Summary_Text", "summary")

# Filter dataset to remove empty examples and very short summaries (likely noise)
def filter_dataset(example):
    return (
        example["document"] and 
        example["summary"] and 
        len(example["document"]) > 100 and 
        len(example["summary"]) > 50 and 
        len(example["document"]) > len(example["summary"])
    )

filtered_dataset = dataset["train"].filter(filter_dataset)
print(f"Filtered from {len(dataset['train'])} to {len(filtered_dataset)} examples")

# Split dataset into training and validation
train_test_split = filtered_dataset.train_test_split(test_size=0.1)

# Load BART tokenizer with special handling for long sequences
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check lengths to set appropriate max_length
document_lengths = [len(tokenizer.encode(doc)) for doc in train_test_split["train"]["document"][:100]]
summary_lengths = [len(tokenizer.encode(summ)) for summ in train_test_split["train"]["summary"][:100]]

print(f"Average document length: {sum(document_lengths)/len(document_lengths)}")
print(f"Average summary length: {sum(summary_lengths)/len(summary_lengths)}")
print(f"Max document length: {max(document_lengths)}")
print(f"Max summary length: {max(summary_lengths)}")

# Improved preprocessing function for better summarization
def improved_preprocess_function(examples):
    # For input: truncate from the beginning to keep conclusion parts
    # Research papers often have the most important content in the abstract and conclusion
    inputs = tokenizer(
        examples["document"],
        padding="max_length",
        truncation=True,
        max_length=1024,  # Can be adjusted based on your GPU memory
    )
    
    # For summaries: don't pad here (we'll do it dynamically in the data collator)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], 
            truncation=True,
            max_length=256,  # Adjust as needed
        )
    
    inputs["labels"] = labels["input_ids"]
    return inputs

# Apply preprocessing to the dataset
tokenized_train_dataset = train_test_split["train"].map(
    improved_preprocess_function,
    batched=True,
    remove_columns=["document", "summary", "Paper_ID"]
)

tokenized_eval_dataset = train_test_split["test"].map(
    improved_preprocess_function,
    batched=True,
    remove_columns=["document", "summary", "Paper_ID"]
)

# Create data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model_name,
    padding=True,
    return_tensors="pt"
)

# Load BART model - quantized for efficiency
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare for better generative performance
model.config.forced_bos_token_id = tokenizer.bos_token_id
model.config.forced_eos_token_id = tokenizer.eos_token_id
model.config.num_beams = 5
model.config.max_length = 256
model.config.min_length = 100  # Enforce minimum summary length
model.config.length_penalty = 1.0
model.config.no_repeat_ngram_size = 3

# Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# Define improved LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=32,  # Higher rank for better representation
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Add more attention modules
)

# Apply LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ROUGE metric for evaluation
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate ROUGE scores
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    result = {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
    }
    
    return result

# Improved training arguments for better performance
training_args = Seq2SeqTrainingArguments(
    output_dir="./improved-bart-lora",
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    learning_rate=5e-5,  # Slightly lower learning rate for better convergence
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,  # Effective batch size = 2*16 = 32
    num_train_epochs=5,  # Train for more epochs
    weight_decay=0.01,
    save_total_limit=3,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=256,
    generation_num_beams=5,
    load_best_model_at_end=True,
    metric_for_best_model="rouge2",  # Use ROUGE-2 as the primary metric
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",  # Disable wandb/tensorboard reporting
)

# Create Seq2Seq trainer with our improved configuration
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
print("Starting training...")
trainer.train()

# Save the model
model.save_pretrained("improved-bart-lora")
tokenizer.save_pretrained("improved-bart-lora")
print("Training completed and model saved to improved-bart-lora/") 