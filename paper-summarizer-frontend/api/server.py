import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import torch
import fitz  # PyMuPDF is imported as fitz
from PIL import Image
import re
import numpy as np

# Define the preprocess_text function here instead of importing it
def preprocess_text(text):
    """Removes author names and unnecessary metadata from research papers."""
    # Check if text is empty
    if not text:
        return ""
    
    # First, split the text to isolate the frontmatter (title, authors, affiliations)
    lines = text.split('\n')
    filtered_lines = []
    skip_until_abstract = True
    
    for line in lines:
        # Look for specific sections that indicate the paper content is starting
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
        
    # Remove references and appendices which often confuse the model
    text = re.sub(r"References\s*\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"Bibliography\s*\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"Appendix\s*\n.*", "", text, flags=re.DOTALL)
    
    # Clean excessive whitespace
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

app = FastAPI(title="Research Paper Summarizer API")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tokenizer and model from the fine-tuned directory
@app.on_event("startup")
async def load_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    global tokenizer, model
    
    # Try several models in order of preference until one loads successfully
    models_to_try = [
        {
            "name": "facebook/bart-large-xsum",
            "tokenizer_cls": AutoTokenizer,
            "model_cls": AutoModelForSeq2SeqLM,
            "description": "BART XSum"
        },
        {
            "name": "facebook/bart-large-cnn",
            "tokenizer_cls": AutoTokenizer,
            "model_cls": AutoModelForSeq2SeqLM,
            "description": "BART CNN"
        },
        {
            "name": "sshleifer/distilbart-cnn-12-6",
            "tokenizer_cls": AutoTokenizer,
            "model_cls": AutoModelForSeq2SeqLM,
            "description": "DistilBART CNN"
        }
    ]
    
    for model_info in models_to_try:
        try:
            print(f"Trying to load {model_info['description']} model ({model_info['name']})...")
            tokenizer = model_info["tokenizer_cls"].from_pretrained(model_info["name"])
            model = model_info["model_cls"].from_pretrained(model_info["name"])
            
            if torch.cuda.is_available():
                model = model.to("cuda")
                print(f"{model_info['description']} model loaded on GPU")
            else:
                print(f"{model_info['description']} model loaded on CPU")
                
            print(f"{model_info['description']} model loaded successfully!")
            # Save model name for future reference
            global MODEL_NAME
            MODEL_NAME = model_info["name"]
            break
        except Exception as e:
            print(f"Error loading {model_info['description']} model: {str(e)}")
    
    if 'model' not in globals():
        raise Exception("Failed to load any summarization model")

def extract_text_from_pdf(pdf_bytes):
    """Extracts text from PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join([page.get_text("text") for page in doc])
    return preprocess_text(full_text) if full_text else None

def extract_images_from_pdf(pdf_bytes):
    """Extracts images and captions from the PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    
    for page_num, page in enumerate(doc):
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            
            # Convert to PIL Image and then to base64 for sending to frontend
            image = Image.open(io.BytesIO(image_data))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Try to extract caption
            caption = f"Figure {img_index+1} (Page {page_num+1})"
            lines = page.get_text("text").split("\n")
            for i, line in enumerate(lines):
                if f"Figure {img_index+1}" in line or "Fig." in line:
                    caption = line
                    break
            
            images.append({
                "image": f"data:image/png;base64,{img_base64}",
                "caption": caption
            })
    
    return images

def remove_author_references(text):
    """Aggressively remove any author pattern references from the summary."""
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
    
    return text

def clean_underscores(text):
    """Remove or fix underscore artifacts that models sometimes produce."""
    # Remove long sequences of underscores (more than 5 consecutive)
    text = re.sub(r'_{5,}', '', text)
    
    # Replace shorter underscore sequences with more appropriate punctuation
    text = re.sub(r'_{2,4}', '. ', text)
    text = re.sub(r'_', ' ', text)  # Replace single underscores with spaces
    
    # Fix spacing after cleanup
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\.\s+\.', '.', text)
    text = re.sub(r'\.\.+', '.', text)
    
    return text

def fix_transformer_names(text):
    """Fix common transformer model names that might be corrupted."""
    model_replacements = {
        'transformer': 'Transformer',
        'bart': 'BART',
        'gpt': 'GPT',
        'bert': 'BERT',
        't5': 'T5',
        'pegasus': 'PEGASUS',
        'roberta': 'RoBERTa',
        'llama': 'LLaMA',
        'long[ -]?former': 'Longformer'
    }
    
    for pattern, replacement in model_replacements.items():
        text = re.sub(rf'\b{pattern}\b', replacement, text, flags=re.IGNORECASE)
    
    return text

def improve_summary(text):
    """Post-processes the summary to improve quality."""
    # Check if the text is None or empty
    if not text:
        return "Could not generate a summary for this paper."
    
    # Clean up underscore artifacts
    text = clean_underscores(text)
    
    # Fix transformer model names
    text = fix_transformer_names(text)
    
    # Remove any specific author references
    text = remove_author_references(text)
    
    # Fix capitalization after removing authors
    text = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), text)
    
    # Ensure proper sentence endings
    if text and not text.endswith(('.', '!', '?')):
        text = text + '.'
    
    # Remove redundant sentences or fragments
    sentences = re.split(r'(?<=[.!?]) +', text)
    unique_sentences = []
    seen_content = set()
    
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < 4:
            continue
            
        # Normalize the sentence for comparison (lowercase, remove punctuation)
        normalized = re.sub(r'[^\w\s]', '', sentence.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # If it's not too similar to something we've seen, keep it
        if normalized and normalized not in seen_content and len(normalized) > 10:
            unique_sentences.append(sentence)
            seen_content.add(normalized)
    
    # If we filtered out too much, return a subset of the original sentences
    if not unique_sentences and sentences:
        return ' '.join(sentences[:3])
    
    result = ' '.join(unique_sentences)
    
    # Final cleanup
    result = re.sub(r'\s+', ' ', result).strip()
    
    # If result is too short or empty, return a generic message
    if len(result.split()) < 10:
        return "This paper introduces a new approach in the field. Please check the full paper for details."
    
    return result

def summarize_text(text):
    """Summarizes text using the loaded model."""
    # Clean and prepare input text
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Special handling for different models
    if 't5' in MODEL_NAME:
        text = "summarize: " + text
    
    # Determine max input tokens based on model
    if 'bart' in MODEL_NAME:
        max_tokens = 1024  # BART typically handles 1024 tokens
    elif 'led' in MODEL_NAME:
        max_tokens = 16384  # LED can handle much longer sequences
    else:
        max_tokens = 512  # Conservative default
    
    # Truncate text if it's too long
    inputs = tokenizer([text], max_length=max_tokens, truncation=True, return_tensors="pt")
    
    # Generate the summary
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Set generation parameters
    summary_ids = model.generate(
        **inputs,
        max_length=256,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Post-process to improve the summary
    summary = improve_summary(summary)
    
    return summary

@app.post("/api/summarize")
async def summarize_paper(file: UploadFile = File(...)):
    """Endpoint to summarize a research paper PDF."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    content = await file.read()
    
    if len(content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="File size must be less than 10MB")
    
    try:
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(content)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF")
        
        # Generate summary
        summary = summarize_text(extracted_text)
        
        # Extract images
        images = extract_images_from_pdf(content)
        
        return {
            "summary": summary,
            "images": images
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 