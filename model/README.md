---
license: apache-2.0
datasets:
- kritsadaK/EDGAR-CORPUS-Financial-Summarization
language:
- en
metrics:
- rouge
base_model:
- facebook/bart-large-cnn
---
# **BART Financial Summarization Model**  

**Model Name:** `kritsadaK/bart-financial-summarization`  
**Base Model:** `facebook/bart-large-cnn`  
**Task:** Financial Text Summarization  
**Dataset:** `kritsadaK/EDGAR-CORPUS-Financial-Summarization`  

**Techniques:**  
- Fine-tuned using the Hugging Face `Trainer` API  
- Tokenized with `AutoTokenizer` (max length 1024 for input, 256 for summary)  
- Optimized with AdamW, learning rate `2e-5`, batch size `2`, `fp16` enabled  
- Evaluated using ROUGE scores  

**Evaluation Results:**  
- **Loss:** 1.18  
- **Runtime:** 18.9 seconds  
- **Samples per second:** 56.1  
- **Steps per second:** 28.1  
- **Epochs:** 3  

**Usage Example (Python):**  
```python
from transformers import pipeline

max_input_length = 1024  
summarizer = pipeline("summarization", model="kritsadaK/bart-financial-summarization")
text = "Your financial document text here..."
summary = summarizer(text, max_length=256, min_length=50, do_sample=False)
print(summary)
```


The **Financial Statements Summary 10K Dataset** was developed as part of the **CSX4210: Natural Language Processing** project at **Assumption University**.