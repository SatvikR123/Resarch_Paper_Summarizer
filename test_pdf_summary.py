#!/usr/bin/env python3
"""
Tool to test PDF summarization with updated parameters
"""
import os
import argparse
import sys
import torch
import fitz  # PyMuPDF
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def preprocess_text(text):
    """Removes author names and unnecessary metadata from research papers."""
    # Remove references and appendices which often confuse the model
    text = re.sub(r"References\s*\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"Bibliography\s*\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"Appendix\s*\n.*", "", text, flags=re.DOTALL)
    
    # Clean author information
    text = re.sub(r"(?i)(?:by|authors?)\s*[:\n].*?\n\n", "", text, flags=re.DOTALL)
    
    # Clean excessive whitespace
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        full_text = "\n".join([page.get_text("text") for page in doc])
        return full_text if full_text else None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def improve_summary(text):
    """Post-processes the summary to improve quality."""
    # Ensure proper sentence endings
    if text and not text.endswith(('.', '!', '?')):
        text = text + '.'
    
    # Remove redundant sentences or fragments
    sentences = re.split(r'(?<=[.!?]) +', text)
    unique_sentences = []
    seen_content = set()
    
    for sentence in sentences:
        # Normalize the sentence for comparison (lowercase, remove punctuation)
        normalized = re.sub(r'[^\w\s]', '', sentence.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # If it's not too similar to something we've seen, keep it
        if normalized and normalized not in seen_content and len(normalized) > 10:
            unique_sentences.append(sentence)
            seen_content.add(normalized)
    
    return ' '.join(unique_sentences)

def summarize_with_original_params(text, tokenizer, model):
    """Generate summary using original parameters."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        **inputs, 
        max_length=256, 
        num_beams=4, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_with_improved_params(text, tokenizer, model):
    """Generate summary using improved parameters."""
    # Clean and prepare input text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate text if it's too long (BART can handle up to 1024 tokens)
    max_chars = 8000  # Approximate character limit (varies by text)
    if len(text) > max_chars:
        text = text[:max_chars]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    
    # Generate with improved parameters
    summary_ids = model.generate(
        **inputs, 
        max_length=256,
        min_length=100,  # Encourage longer summaries
        num_beams=5,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
        do_sample=True,  # Enable sampling for more creative summaries
        top_k=50,
        top_p=0.9,
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Post-process to improve the summary
    summary = improve_summary(summary)
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Test PDF summarization with updated parameters")
    parser.add_argument("pdf_path", help="Path to the PDF file to summarize")
    parser.add_argument("--model_path", default="fine-tuned-bart-lora", 
                        help="Path to the fine-tuned model directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found at {args.pdf_path}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Extract text from PDF
    print(f"Extracting text from {args.pdf_path}...")
    pdf_text = extract_text_from_pdf(args.pdf_path)
    
    if not pdf_text:
        print("Failed to extract text from PDF.")
        sys.exit(1)
    
    # Preprocess text
    processed_text = preprocess_text(pdf_text)
    
    # Generate summaries with both parameter sets
    print("\nGenerating summary with original parameters...")
    original_summary = summarize_with_original_params(processed_text, tokenizer, model)
    
    print("Generating summary with improved parameters...")
    improved_summary = summarize_with_improved_params(processed_text, tokenizer, model)
    
    # Display results
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print("\nORIGINAL PARAMETERS SUMMARY:")
    print("-"*80)
    print(original_summary)
    print("-"*80)
    print(f"Length: {len(original_summary)} characters")
    
    print("\nIMPROVED PARAMETERS SUMMARY:")
    print("-"*80)
    print(improved_summary)
    print("-"*80)
    print(f"Length: {len(improved_summary)} characters")
    
    # Basic analysis
    print("\nANALYSIS:")
    print("-"*80)
    length_diff = len(improved_summary) - len(original_summary)
    print(f"Length difference: {length_diff} characters")
    
    original_sentences = len(re.split(r'(?<=[.!?]) +', original_summary))
    improved_sentences = len(re.split(r'(?<=[.!?]) +', improved_summary))
    print(f"Original summary: {original_sentences} sentences")
    print(f"Improved summary: {improved_sentences} sentences")
    
    # Check for duplicate sentences
    original_sentences_list = re.split(r'(?<=[.!?]) +', original_summary)
    orig_normalized = [re.sub(r'[^\w\s]', '', s.lower()).strip() for s in original_sentences_list]
    orig_duplicates = len(orig_normalized) - len(set(orig_normalized))
    
    improved_sentences_list = re.split(r'(?<=[.!?]) +', improved_summary)
    impr_normalized = [re.sub(r'[^\w\s]', '', s.lower()).strip() for s in improved_sentences_list]
    impr_duplicates = len(impr_normalized) - len(set(impr_normalized))
    
    print(f"Original summary has {orig_duplicates} duplicate sentences")
    print(f"Improved summary has {impr_duplicates} duplicate sentences")

if __name__ == "__main__":
    main() 