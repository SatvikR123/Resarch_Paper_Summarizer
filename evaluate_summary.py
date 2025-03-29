"""
Script to evaluate and compare summarization quality between original and improved models.
"""
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import fitz  # PyMuPDF
import re
import argparse
from rouge_score import rouge_scorer

# Original model preprocessing
def preprocess_text_original(text):
    """Original preprocessing function."""
    text = re.sub(r"(?i)(?:by|authors?)\s*[:\n].*?\n\n", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text

# Improved model preprocessing 
def preprocess_text_improved(text):
    """Improved preprocessing for better summaries."""
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
    """Extract text from PDF file."""
    try:
        doc = fitz.open(pdf_path)
        full_text = "\n".join([page.get_text("text") for page in doc])
        return full_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def get_original_summary(text, tokenizer, model):
    """Generate summary using the original model settings."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        **inputs, 
        max_length=256, 
        num_beams=4, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def improve_summary(text):
    """Post-process the summary to enhance quality."""
    # Ensure proper sentence endings
    if text and not text.endswith(('.', '!', '?')):
        text = text + '.'
    
    # Remove redundant sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    unique_sentences = []
    seen_content = set()
    
    for sentence in sentences:
        normalized = re.sub(r'[^\w\s]', '', sentence.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        if normalized and normalized not in seen_content and len(normalized) > 10:
            unique_sentences.append(sentence)
            seen_content.add(normalized)
    
    return ' '.join(unique_sentences)

def get_improved_summary(text, tokenizer, model):
    """Generate summary with improved parameters."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    
    summary_ids = model.generate(
        **inputs, 
        max_length=256,
        min_length=100,
        num_beams=5,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return improve_summary(summary)

def evaluate_summaries(original_summary, improved_summary):
    """Compare summaries using ROUGE metrics."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Use each summary as reference for the other to compare differences
    scores_original_ref = scorer.score(original_summary, improved_summary)
    scores_improved_ref = scorer.score(improved_summary, original_summary)
    
    # Calculate diversity score (lower means more diverse summaries)
    diversity = sum([
        scores_original_ref['rouge1'].fmeasure,
        scores_original_ref['rouge2'].fmeasure,
        scores_original_ref['rougeL'].fmeasure
    ]) / 3.0
    
    # Analyze length difference (sign indicates which is longer)
    length_diff = len(improved_summary) - len(original_summary)
    length_ratio = len(improved_summary) / len(original_summary) if len(original_summary) > 0 else float('inf')
    
    return {
        "diversity_score": diversity,
        "length_difference": length_diff,
        "length_ratio": length_ratio,
        "original_length": len(original_summary),
        "improved_length": len(improved_summary),
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare summarization models")
    parser.add_argument("pdf_path", help="Path to PDF file to summarize")
    parser.add_argument("--model_path", default="fine-tuned-bart-lora", 
                        help="Path to fine-tuned model")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found at {args.pdf_path}")
        sys.exit(1)
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    print(f"Extracting text from {args.pdf_path}...")
    pdf_text = extract_text_from_pdf(args.pdf_path)
    
    if not pdf_text:
        print("Failed to extract text from PDF.")
        sys.exit(1)
    
    # Process text with different methods
    original_processed = preprocess_text_original(pdf_text)
    improved_processed = preprocess_text_improved(pdf_text)
    
    print("\nGenerating original summary...")
    original_summary = get_original_summary(original_processed, tokenizer, model)
    
    print("Generating improved summary...")
    improved_summary = get_improved_summary(improved_processed, tokenizer, model)
    
    # Evaluate the differences
    print("\nEvaluating summaries...")
    metrics = evaluate_summaries(original_summary, improved_summary)
    
    # Display results
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print("\nORIGINAL SUMMARY:")
    print("-"*80)
    print(original_summary)
    print("-"*80)
    print(f"Length: {metrics['original_length']} characters")
    
    print("\nIMPROVED SUMMARY:")
    print("-"*80)
    print(improved_summary)
    print("-"*80)
    print(f"Length: {metrics['improved_length']} characters")
    
    print("\nMETRICS:")
    print("-"*80)
    print(f"Diversity score: {metrics['diversity_score']:.4f} (lower means more diverse)")
    print(f"Length difference: {metrics['length_difference']} characters")
    print(f"Length ratio (improved/original): {metrics['length_ratio']:.2f}")
    
    # Simple analysis
    print("\nANALYSIS:")
    print("-"*80)
    
    if metrics['diversity_score'] < 0.7:
        print("✓ The improved summary differs significantly from the original")
    else:
        print("⚠ The improved summary is still quite similar to the original")
        
    if metrics['length_ratio'] > 1.2:
        print("✓ The improved summary is more comprehensive")
    elif metrics['length_ratio'] < 0.8:
        print("⚠ The improved summary is much shorter than the original")
    else:
        print("✓ The summaries have comparable length")
    
    # Look for duplicate sentences in original summary
    original_sentences = re.split(r'(?<=[.!?]) +', original_summary)
    orig_normalized = [re.sub(r'[^\w\s]', '', s.lower()).strip() for s in original_sentences]
    orig_duplicates = len(orig_normalized) - len(set(orig_normalized))
    
    improved_sentences = re.split(r'(?<=[.!?]) +', improved_summary)
    impr_normalized = [re.sub(r'[^\w\s]', '', s.lower()).strip() for s in improved_sentences]
    impr_duplicates = len(impr_normalized) - len(set(impr_normalized))
    
    print(f"✓ Original summary has {orig_duplicates} duplicate sentences")
    print(f"✓ Improved summary has {impr_duplicates} duplicate sentences")
    
    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    if metrics['diversity_score'] < 0.7 and metrics['length_ratio'] > 1.0 and impr_duplicates < orig_duplicates:
        print("✓ The improved summary appears to be better quality")
    elif metrics['diversity_score'] > 0.9:
        print("⚠ Minimal difference between the two approaches")
    else:
        print("⚠ Mixed results - manual review recommended")

if __name__ == "__main__":
    main() 