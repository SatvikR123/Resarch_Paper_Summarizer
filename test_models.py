#!/usr/bin/env python3
"""
Test script to compare different summarization models with research papers.
This script doesn't need any external API keys and uses open-source models.
"""
import argparse
import re
import torch
from transformers import (
    PegasusForConditionalGeneration, PegasusTokenizer,
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    LEDForConditionalGeneration, LEDTokenizer
)
import fitz  # PyMuPDF

# Dictionary of available models to test
AVAILABLE_MODELS = {
    "pegasus-arxiv": {
        "name": "google/pegasus-arxiv",
        "tokenizer_class": PegasusTokenizer,
        "model_class": PegasusForConditionalGeneration,
        "max_length": 256,
        "min_length": 100,
        "description": "PEGASUS model fine-tuned on arXiv dataset (scientific papers)"
    },
    "pegasus-pubmed": {
        "name": "google/pegasus-pubmed",
        "tokenizer_class": PegasusTokenizer,
        "model_class": PegasusForConditionalGeneration,
        "max_length": 256,
        "min_length": 100,
        "description": "PEGASUS model fine-tuned on PubMed dataset (medical papers)"
    },
    "led-large-16384": {
        "name": "allenai/led-large-16384",
        "tokenizer_class": LEDTokenizer,
        "model_class": LEDForConditionalGeneration,
        "max_length": 256,
        "min_length": 100,
        "description": "Longformer Encoder-Decoder model for long document summarization"
    },
    "bart-large-cnn": {
        "name": "facebook/bart-large-cnn",
        "tokenizer_class": BartTokenizer,
        "model_class": BartForConditionalGeneration,
        "max_length": 256,
        "min_length": 50,
        "description": "BART model fine-tuned on CNN news articles"
    },
    "t5-small": {
        "name": "t5-small",
        "tokenizer_class": T5Tokenizer,
        "model_class": T5ForConditionalGeneration,
        "max_length": 256,
        "min_length": 50,
        "description": "Small T5 model (can be used for summarization with 'summarize:' prefix)"
    }
}

def preprocess_text(text):
    """Preprocess text for summarization."""
    # Split by lines to handle frontmatter
    lines = text.split('\n')
    filtered_lines = []
    skip_until_abstract = True
    
    for line in lines:
        # Start including content after abstract or introduction
        if skip_until_abstract and any(marker in line.lower() for marker in 
                                     ['abstract', 'introduction', '1. introduction', '1 introduction']):
            skip_until_abstract = False
        
        # Skip lines until we reach substantive content
        if skip_until_abstract:
            # Skip lines that match author patterns
            if re.search(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', line):  # First Last name pattern
                continue
            if re.search(r'^\s*[A-Z][a-z]+,\s+[A-Z][a-z]+', line):  # Last, First pattern
                continue
            if re.search(r'University|Institute|School|College|Laboratory|Corporation|Corp\.', line):
                continue
            if re.search(r'\w+@\w+\.\w+', line):  # Email pattern
                continue
            if re.search(r'^\s*[0-9]', line):  # Lines starting with numbers (often footnotes)
                continue
        
        # Add the line if it passed all filters
        filtered_lines.append(line)
    
    # Rejoin the filtered lines
    text = '\n'.join(filtered_lines)
        
    # Remove references and appendices
    text = re.sub(r"References\s*\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"Bibliography\s*\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"Appendix\s*\n.*", "", text, flags=re.DOTALL)
    
    # Clean whitespace
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_author_references(text):
    """Post-process to remove author references."""
    # Replace patterns like "Author et al." with "This paper"
    text = re.sub(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+et\s+al\.', 'This paper', text)
    
    # Replace full author lists with "The authors"
    text = re.sub(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,|\sand|\s&)?){2,}', 'The authors', text)
    
    # Replace specific verbs commonly used with author references
    for verb in ['present', 'propose', 'describe', 'introduce', 'report', 'demonstrate']:
        # Patterns like "Author presents" or "Authors present"
        text = re.sub(rf'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+{verb}s?', f'This paper {verb}s', text)
        text = re.sub(rf'The authors {verb}s?', f'This paper {verb}s', text)
    
    # Remove academic paper title
    text = re.sub(r'^"[^"]+"\s+', '', text)
    
    # Replace all author references
    text = re.sub(r'\b(?:[A-Z][a-z]*\.?\s*)+(?:and|&)\s+(?:[A-Z][a-z]*\.?\s*)+', 'researchers', text)
    
    # Remove other references to "our" or "we"
    text = re.sub(r'\bOur\b', 'The', text)
    text = re.sub(r'\bour\b', 'the', text)
    text = re.sub(r'\bWe\b', 'The researchers', text)
    text = re.sub(r'\bwe\b', 'the researchers', text)
    
    # Fix capitalization after removing authors
    text = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), text)
    
    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def load_model(model_key):
    """Load a model by its key."""
    if model_key not in AVAILABLE_MODELS:
        print(f"Model {model_key} not available.")
        print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return None, None
    
    model_info = AVAILABLE_MODELS[model_key]
    print(f"Loading {model_info['name']}...")
    
    try:
        tokenizer = model_info['tokenizer_class'].from_pretrained(model_info['name'])
        model = model_info['model_class'].from_pretrained(model_info['name'])
        
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
            
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def summarize_with_model(text, model_key, model=None, tokenizer=None):
    """Generate a summary using the specified model."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_key)
        if model is None or tokenizer is None:
            return None
    
    # Clean and prepare text
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Special handling for T5 models
    if 't5' in model_key:
        # T5 requires a prefix for the task
        text = "summarize: " + text
    
    # Handle different model input requirements
    if 'led' in model_key:
        # LED models handle longer sequences
        max_tokens = 16384
    else:
        max_tokens = 1024
    
    # Encode the text
    inputs = tokenizer([text], max_length=max_tokens, truncation=True, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate the summary
    summary_ids = model.generate(
        **inputs,
        max_length=AVAILABLE_MODELS[model_key]['max_length'],
        min_length=AVAILABLE_MODELS[model_key]['min_length'],
        length_penalty=2.0,
        num_beams=6,
        early_stopping=True
    )
    
    # Decode and post-process
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = remove_author_references(summary)
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Test different summarization models with research papers")
    parser.add_argument("pdf_path", help="Path to the PDF file to summarize")
    parser.add_argument("--models", nargs="+", default=["pegasus-arxiv"], 
                        help="Models to test for summarization")
    parser.add_argument("--list-models", action="store_true", 
                        help="List available models and exit")
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for key, model_info in AVAILABLE_MODELS.items():
            print(f"- {key}: {model_info['description']}")
        return
    
    # Extract text from PDF
    text = extract_text_from_pdf(args.pdf_path)
    if not text:
        print("Failed to extract text from PDF.")
        return
    
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Test each requested model
    for model_key in args.models:
        if model_key not in AVAILABLE_MODELS:
            print(f"Model {model_key} not available. Skipping.")
            continue
        
        print(f"\n{'-'*80}")
        print(f"TESTING MODEL: {model_key}")
        print(f"Description: {AVAILABLE_MODELS[model_key]['description']}")
        print(f"{'-'*80}")
        
        summary = summarize_with_model(preprocessed_text, model_key)
        
        if summary:
            print("\nSUMMARY:")
            print(f"{'-'*80}")
            print(summary)
            print(f"{'-'*80}")
            print(f"Length: {len(summary.split())} words")
        else:
            print("Failed to generate summary.")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 