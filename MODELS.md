# Research Paper Summarization Models

This document outlines the different summarization models available in this project and how to use them.

## Available Models

We've implemented multiple summarization models that don't require any API keys:

1. **PEGASUS ArXiv** (Default in the API)
   - Specifically fine-tuned on scientific papers from ArXiv
   - Generally produces the best results for research papers
   - Better at avoiding author names and direct copying

2. **PEGASUS PubMed**
   - Fine-tuned on medical research papers
   - Good for biomedical or healthcare papers

3. **Longformer Encoder-Decoder (LED)**
   - Can handle much longer documents (up to 16K tokens)
   - Good for very detailed papers

4. **BART CNN**
   - General summarization model fine-tuned on news articles
   - Our original fine-tuned model was based on this

5. **T5**
   - General-purpose text-to-text model
   - Can be used for various NLP tasks including summarization

## Using the Models

### Through the Web Interface

The web interface now uses the PEGASUS ArXiv model by default. Just upload your PDF as usual.

### Testing Different Models

To try different models with the same paper, you can use the `test_models.py` script:

```bash
# Show available models
python test_models.py --list-models

# Test a specific model
python test_models.py path/to/your/paper.pdf --models pegasus-arxiv

# Test multiple models at once
python test_models.py path/to/your/paper.pdf --models pegasus-arxiv bart-large-cnn t5-small
```

### Modifying the API

To change which model the API uses, edit the `server.py` file:

1. Open `paper-summarizer-frontend/api/server.py`
2. In the `load_model` function, change the `model_name` variable to your preferred model:
   ```python
   # Use the PEGASUS model fine-tuned on scientific papers
   model_name = "google/pegasus-arxiv"  # Change this to another model
   ```
3. Restart the server

## Tips for Better Summaries

1. **PDF Quality**: Make sure your PDF has selectable text (not just scanned images)
2. **Paper Length**: The model works best on papers under 10-15 pages
3. **Scientific Domain**: For papers in specialized domains, try domain-specific models:
   - Medical papers: Use `pegasus-pubmed`
   - Very long papers: Use `led-large-16384`

## Troubleshooting

If you encounter issues with summaries:

1. **Author Names**: If author names still appear, our automatic filtering might not catch all patterns. You can manually edit the summary afterward.

2. **Abstract Copying**: If the summary still copies from the abstract, try using a different model. The PEGASUS models tend to be more abstractive.

3. **Out of Memory Errors**: Some models (especially LED) require more memory. Try using a smaller model like `bart-large-cnn` or `t5-small`.

4. **Text Extraction Issues**: If the text extraction from PDF fails, try converting your PDF to text using another tool before summarizing.

## Further Improvement

For further model improvement without retraining:

1. Use the `test_models.py` script to find the best model for your specific papers
2. Try different preprocessing approaches (you can modify the `preprocess_text` function)
3. Edit the post-processing (modify the `remove_author_references` function) 