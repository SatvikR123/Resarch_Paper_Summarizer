#!/usr/bin/env python3
"""
Quick test to compare the original and enhanced model summaries
without running the full server.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import sys

# Sample text from a research paper (abstract or short segment)
SAMPLE_TEXT = """
Automatic summarization has traditionally relied on human-written reference 
summaries for evaluation and training. In this work, we show that we can
achieve state-of-the-art results on summarization without using any 
human-written summaries at all. We introduce a method of training
summarization models based on question-answering, which does not require 
reference summaries. Our models achieve the state-of-the-art on the
abstractive CNN/DailyMail task, outperforming pointer-generator models
and even models that use the reference summaries.
"""

def clean_text(text):
    """Basic text cleaning for test"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_summary(model_path, text):
    """Generate a summary using the specified model"""
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Set generation parameters
        model.config.num_beams = 5  # More beams for better quality
        model.config.length_penalty = 1.0  # Encourage slightly longer summaries
        model.config.no_repeat_ngram_size = 3  # Avoid repetition
        model.config.early_stopping = True
        
        # Clean input text
        text = clean_text(text)
        
        # Tokenize and generate
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        
        # Generate with improved parameters
        summary_ids = model.generate(
            **inputs, 
            max_length=256,
            min_length=50,  # Ensure minimum length
            num_beams=5,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=True,  # Enable sampling for more creative summaries
            top_k=50,
            top_p=0.9,
        )
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # Use sample text or a text from command line
    if len(sys.argv) > 1:
        if sys.argv[1].endswith('.txt'):
            try:
                with open(sys.argv[1], 'r') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                text = SAMPLE_TEXT
        else:
            text = sys.argv[1]
    else:
        text = SAMPLE_TEXT
        
    # Generate summaries using both models
    print("Using original model...")
    original_summary = generate_summary("fine-tuned-bart-lora", text)
    
    print("\nUsing enhanced model...")
    enhanced_summary = generate_summary("enhanced-bart-lora", text)
    
    # Print the results
    print("\n" + "="*80)
    print("COMPARISON OF SUMMARIES")
    print("="*80)
    
    print("\nINPUT TEXT:")
    print("-"*80)
    print(text[:300] + "..." if len(text) > 300 else text)
    print("-"*80)
    
    print("\nORIGINAL MODEL SUMMARY:")
    print("-"*80)
    print(original_summary if original_summary else "Failed to generate summary")
    print("-"*80)
    
    print("\nENHANCED MODEL SUMMARY:")
    print("-"*80)
    print(enhanced_summary if enhanced_summary else "Failed to generate summary")
    print("-"*80)
    
    # Simple analysis
    if original_summary and enhanced_summary:
        original_words = len(original_summary.split())
        enhanced_words = len(enhanced_summary.split())
        
        print("\nANALYSIS:")
        print(f"Original summary: {original_words} words")
        print(f"Enhanced summary: {enhanced_words} words")
        print(f"Word count difference: {enhanced_words - original_words} words")
        
        if enhanced_words > original_words:
            print("✓ Enhanced model produced a longer summary")
        else:
            print("⚠ Enhanced model produced a shorter summary")

if __name__ == "__main__":
    main() 