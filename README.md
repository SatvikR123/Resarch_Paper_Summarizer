<h1 align="center" id="title">Research Paper Summarizer</h1>

<p align="center"><img src="https://socialify.git.ci/SatvikR123/Resarch_Paper_Summarizer/image?font=Raleway&amp;language=1&amp;name=1&amp;owner=1&amp;pattern=Brick+Wall&amp;theme=Auto" alt="project-image"></p>

<p id="description">An AI-powered Research Paper Summarizer that extracts key insights from academic papers providing a concise summary and relevant images. This tool leverages NLP models to process PDF research papers efficiently.</p>


## 📂 Project Structure  

```
📁 Research_Paper_Summarizer  
 ├── 📁 Papers/                 # Folder containing sample research papers  
 ├── 📁 fine-tuned-bart-lora/   # Model weights for fine-tuned BART  
 ├── 📄 app.py                  # Streamlit application  
 ├── 📄 model.py                # Loads and processes the AI model  
 ├── 📄 createDataset.py        # Script for dataset processing  
 ├── 📄 extracted_data.csv      # Extracted text dataset  
 ├── 📄 README.md               # Project documentation 
```
  
<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Summarizes research papers using a fine-tuned BART model
*   Extracts key figures and images from the PDF
*   Easy-to-use interface with Streamlit

<h2>🛠️ Installation Steps:</h2>

<p>1. Clone the Repository</p>

```
git clone https://github.com/SatvikR123/Research_Paper_Summarizer.git cd Research_Paper_Summarizer
```

<p>2. Install Dependencies</p>

```
pip install -r requirements.txt
```

<p>3. Run the Application</p>

```
streamlit run app.py
```

<h2>🍰 Contribution Guidelines:</h2>

Contributions are welcome! If you have ideas to improve this project feel free to open an issue or submit a pull request.

  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Python
*   Streamlit
*   Transformers
*   Pytorch
