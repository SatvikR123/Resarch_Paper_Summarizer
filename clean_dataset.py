#!/usr/bin/env python3
"""
Script to clean and enhance the research paper dataset for better fine-tuning.
"""
import pandas as pd
import re
import argparse
import nltk
import os
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """Clean and normalize research paper text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove references section
    text = re.sub(r"References\s*\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"Bibliography\s*\n.*", "", text, flags=re.DOTALL)
    
    # Remove author information
    text = re.sub(r"(?i)(?:by|authors?)\s*[:\n].*?\n\n", "", text, flags=re.DOTALL)
    
    # Clean excessive whitespace
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_abstract_in_text(summary, text):
    """Check if the summary is directly copied from the text."""
    # Normalize both for comparison
    norm_summary = re.sub(r'\s+', ' ', summary.lower()).strip()
    norm_text = re.sub(r'\s+', ' ', text.lower()).strip()
    
    # Check if summary appears verbatim in text
    return norm_summary in norm_text

def clean_summary(summary):
    """Clean and improve summary text."""
    if not isinstance(summary, str) or not summary.strip():
        return ""
    
    # Normalize whitespace
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    # Ensure proper sentence endings
    if summary and not summary.endswith(('.', '!', '?')):
        summary = summary + '.'
    
    # Remove duplicated sentences
    sentences = sent_tokenize(summary)
    unique_sentences = []
    seen_content = set()
    
    for sentence in sentences:
        # Normalize for comparison
        normalized = re.sub(r'[^\w\s]', '', sentence.lower()).strip()
        
        if normalized and normalized not in seen_content:
            unique_sentences.append(sentence)
            seen_content.add(normalized)
    
    return ' '.join(unique_sentences)

def main():
    parser = argparse.ArgumentParser(description="Clean and enhance research paper dataset")
    parser.add_argument("--input_file", default="extracted_data.csv",
                        help="Path to the input CSV file")
    parser.add_argument("--output_file", default="cleaned_dataset.csv",
                        help="Path to save the cleaned dataset")
    parser.add_argument("--min_document_length", type=int, default=200,
                        help="Minimum document length in characters")
    parser.add_argument("--min_summary_length", type=int, default=50,
                        help="Minimum summary length in characters")
    parser.add_argument("--max_abstract_similarity", type=float, default=0.8,
                        help="Max similarity between summary and text to avoid abstracts")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
        return
    
    print(f"Loading dataset from {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file)
        original_count = len(df)
        print(f"Original dataset contains {original_count} examples")
        
        # Rename columns if needed
        if "XML_Text" in df.columns and "document" not in df.columns:
            df = df.rename(columns={"XML_Text": "document"})
        if "Summary_Text" in df.columns and "summary" not in df.columns:
            df = df.rename(columns={"Summary_Text": "summary"})
        
        if "document" not in df.columns or "summary" not in df.columns:
            print("Error: Dataset must contain 'document' and 'summary' columns (or XML_Text and Summary_Text)")
            return
        
        print("Cleaning documents...")
        tqdm.pandas()
        df["document"] = df["document"].progress_apply(clean_text)
        
        print("Cleaning summaries...")
        df["summary"] = df["summary"].progress_apply(clean_summary)
        
        # Filter out examples that don't meet criteria
        print("Filtering dataset...")
        df = df[df["document"].str.len() >= args.min_document_length]
        df = df[df["summary"].str.len() >= args.min_summary_length]
        
        # Filter out examples where document length is less than summary length
        # (These are likely errors or bad examples)
        df = df[df["document"].str.len() > df["summary"].str.len()]
        
        # Check for direct abstract copying (time-consuming process)
        print("Checking for direct abstract copying...")
        abstract_check_results = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            is_abstract = is_abstract_in_text(row["summary"], row["document"])
            abstract_check_results.append(is_abstract)
        
        df["is_abstract"] = abstract_check_results
        
        # Only keep examples where summary doesn't appear to be a direct copy of text
        # This helps remove examples where summary is just the abstract
        print("Removing direct copies of abstracts...")
        filtered_df = df[~df["is_abstract"]]
        
        # If we've filtered out too many, keep some of the better ones
        if len(filtered_df) < original_count * 0.5:
            print("Warning: Too many examples filtered. Keeping some abstract-like summaries.")
            # Sort by length difference (larger difference between document and summary is better)
            df["length_diff"] = df["document"].str.len() - df["summary"].str.len()
            df = df.sort_values("length_diff", ascending=False)
            # Take the top 75% of examples
            keep_count = int(original_count * 0.75)
            filtered_df = df.head(keep_count)
        
        # Drop helper columns
        filtered_df = filtered_df.drop(columns=["is_abstract", "length_diff"], errors="ignore")
        
        # Save cleaned dataset
        print(f"Saving cleaned dataset to {args.output_file}...")
        filtered_df.to_csv(args.output_file, index=False)
        
        print(f"Done! Original dataset: {original_count} examples, Cleaned dataset: {len(filtered_df)} examples")
        print(f"Removed {original_count - len(filtered_df)} problematic examples")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    main() 