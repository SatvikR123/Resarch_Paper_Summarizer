# Enhanced BART Model for Research Paper Summarization

This model is based on the original fine-tuned BART model with improved generation parameters.

## Improvements

- Increased beam search from 4 to 5 beams for more diverse candidate generations
- Added length penalty (1.0) to encourage more informative summaries
- Set minimum summary length to 100 tokens
- Added n-gram repetition penalty to avoid redundant phrases
- Added explicit BOS and EOS token management
- Improved preprocessing and post-processing

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("enhanced-bart-lora")
model = AutoModelForSeq2SeqLM.from_pretrained("enhanced-bart-lora")

# For better generation results:
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
summary_ids = model.generate(
    **inputs, 
    max_length=256,
    min_length=100,
    num_beams=5,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    early_stopping=True,
    do_sample=True,  # Enable sampling for more creative summaries
    top_k=50,
    top_p=0.9,
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

## Original Model

This model is an enhanced version of the model from: fine-tuned-bart-lora
