import streamlit as st
import spacy
import pandas as pd
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import subprocess
import os
os.environ["TRANSFORMERS_CACHE"] = "/home/user/.cache/huggingface"
os.environ["HF_HOME"] = "/home/user/.cache/huggingface"
os.environ["TORCH_HOME"] = "/home/user/.cache/torch"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import nltk
from nltk.tokenize import sent_tokenize
import traceback 
from collections import Counter

# Set Streamlit page config
st.set_page_config(page_title="FinBrief: Financial Document Insights", layout="wide")

try:
    nlp = spacy.load("en_core_web_sm")
    st.write("spaCy model loaded successfully!")
    print("spaCy model loaded successfully!")
except OSError:
    st.write("Failed to load spaCy model. Attempting to install...")
    print("Failed to load spaCy model. Attempting to install...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    try:
        nlp = spacy.load("en_core_web_sm")
        st.write("spaCy model installed and loaded successfully!")
        print("spaCy model installed and loaded successfully!")
    except Exception as e:
        st.write(f"Still failed to load spaCy model: {e}")
        print(f"Still failed to load spaCy model: {e}")
        nlp = None  # Mark spaCy as failed

model_name = "kritsadaK/bart-financial-summarization"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    st.write("Hugging Face summarization model loaded successfully!")
    print("Hugging Face summarization model loaded successfully!")
except Exception as e:
    st.write(f"Failed to load Hugging Face summarization model: {e}")
    print(f"Failed to load Hugging Face summarization model: {e}")
    summarizer = None  # Mark Hugging Face model as failed

# Store models in Streamlit session state
st.session_state["nlp"] = nlp
st.session_state["summarizer"] = summarizer

# UI: Show clear error messages if models failed
if nlp is None:
    st.error("The spaCy model failed to load. Ensure it is installed.")
if summarizer is None:
    st.error("The summarization model failed to load. Check the model path or internet connection.")

st.title("FinBrief: Financial Document Insights")
st.write("Upload a financial document for analysis.")

# Initialize session state
if "nlp" not in st.session_state:
    st.session_state["nlp"] = nlp
if "summarizer" not in st.session_state:
    st.session_state["summarizer"] = summarizer

# Set up NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

def download_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
        st.write("NLTK 'punkt' tokenizer is already installed.")
        print("NLTK 'punkt' tokenizer is already installed.")
    except LookupError:
        st.write("NLTK 'punkt' tokenizer not found. Attempting to download...")
        print("NLTK 'punkt' tokenizer not found. Attempting to download...")
        try:
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
            nltk.data.find('tokenizers/punkt')
            st.write("NLTK 'punkt' tokenizer downloaded successfully.")
            print("NLTK 'punkt' tokenizer downloaded successfully.")
        except Exception as e:
            st.error(f"NLTK 'punkt' tokenizer download failed: {e}")
            print(f"NLTK 'punkt' tokenizer download failed: {e}")

# Call the function at the beginning of script
download_nltk_punkt()

# Debugging: Check session state initialization
print(f"Session State - NLP: {st.session_state['nlp'] is not None}, Summarizer: {st.session_state['summarizer'] is not None}")

# Define regex patterns to extract structured data
patterns = {
    "Fund Name": r"^(.*?) Fund",  # Extracts the name before "Fund"
    "CUSIP": r"CUSIP\s+(\d+)",
    "Inception Date": r"Inception Date\s+([\w\s\d]+)",
    "Benchmark": r"Benchmark\s+([\w\s\d]+)",
    "Expense Ratio": r"Expense Information.*?(\d+\.\d+%)",
    "Total Assets": r"Total Assets\s+USD\s+([\d,]+)",
    "Portfolio Turnover": r"Portfolio Holdings Turnover.*?(\d+\.\d+%)",
    "Cash Allocation": r"% of Portfolio in Cash\s+(\d+\.\d+%)",
    "Alpha": r"Alpha\s+(-?\d+\.\d+%)",
    "Standard Deviation": r"Standard Deviation\s+(\d+\.\d+%)"
}

# Set the title and layout
st.markdown("[Example Financial Documents](https://drive.google.com/drive/folders/1jMu3S7S_Hc_RgK6_cvsCqIB8x3SSS-R6)")

# Custom styling (this remains unchanged)
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f7f7f7;
        color: #333;
    }
    .css-1d391kg {
        background-color: #f0f4f8;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
    }
    .stTextArea textarea {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to extract text and tables using pdfplumber
def extract_text_tables_pdfplumber(pdf_file):
    import io
    import pdfplumber

    print("\nPDFPlumber: Extracting text and tables...")
    with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
        all_text = ""
        all_tables = []

        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"

            # Extract tables
            tables = page.extract_tables()
            all_tables.extend(tables)  # Store all tables

        if all_text.strip():
            print(all_text[:1000])  # Print first 1000 characters for verification
            return all_text, all_tables
        else:
            print("No text extracted. The PDF might be image-based.")
            return None, None

# NEW: Function to evaluate chunk relevance
def evaluate_chunk_relevance(chunk, keywords=None):
    """
    Evaluate the relevance of a text chunk based on various factors.
    Returns a score representing the chunk's relevance.
    """
    if not keywords:
        # Default financial keywords
        keywords = ["fund", "portfolio", "performance", "return", "asset", "investment", 
                    "expense", "risk", "benchmark", "allocation", "strategy", "market",
                    "growth", "income", "dividend", "yield", "capital", "equity", "bond",
                    "summary", "overview", "highlight", "key", "important", "significant"]
    
    score = 0
    
    # Factor 1: Length of the chunk (longer chunks often contain more information)
    word_count = len(chunk.split())
    score += min(word_count / 100, 5)  # Cap at 5 points
    
    # Factor 2: Keyword presence
    # Count keywords in lowercase text
    lower_chunk = chunk.lower()
    keyword_count = sum(1 for keyword in keywords if keyword.lower() in lower_chunk)
    keyword_density = keyword_count / max(1, word_count) * 100
    score += min(keyword_density * 2, 10)  # Cap at 10 points
    
    # Factor 3: Presence of numbers (financial documents often contain important numbers)
    number_count = len(re.findall(r'\d+\.?\d*%?', chunk))
    score += min(number_count / 5, 5)  # Cap at 5 points
    
    # Factor 4: Structured information (lists, tables, etc.)
    bullet_count = len(re.findall(r'•|\*|-|–|[0-9]+\.', chunk))
    score += min(bullet_count, 5)  # Cap at 5 points
    
    # Factor 5: Presence of section headers
    header_patterns = [
        r'^[A-Z][A-Za-z\s]+:',  # Title followed by colon
        r'^[A-Z][A-Z\s]+',      # ALL CAPS text
        r'^\d+\.\s+[A-Z]'       # Numbered section
    ]
    header_count = sum(1 for pattern in header_patterns if re.search(pattern, chunk, re.MULTILINE))
    score += min(header_count * 2, 5)  # Cap at 5 points
    
    return score

# NEW: Function to rank and select the best chunks
def rank_and_select_chunks(chunks, max_chunks=5, keywords=None):
    """
    Rank chunks by relevance and return the top chunks.
    """
    # Evaluate each chunk
    chunk_scores = [(chunk, evaluate_chunk_relevance(chunk, keywords)) for chunk in chunks]
    
    # Sort chunks by score (highest first)
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
    
    # Select the top N chunks
    top_chunks = [chunk for chunk, score in sorted_chunks[:max_chunks]]
    
    # Print scores for debugging
    print("Chunk scores:")
    for i, (chunk, score) in enumerate(sorted_chunks):
        print(f"Chunk {i+1}: Score {score:.2f}, Length {len(chunk.split())} words")
        print(f"First 100 chars: {chunk[:100]}...")
    
    return top_chunks

def split_text_into_chunks(text, tokenizer, max_tokens=512):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ''
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_tokens)

        # If adding the next sentence exceeds the max_tokens limit
        if current_length + sentence_length > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start a new chunk
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += ' ' + sentence
            current_length += sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def remove_duplicate_sentences(text):
    sentences = nltk.sent_tokenize(text)
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        # Normalize the sentence to ignore case and punctuation for comparison
        normalized_sentence = sentence.strip().lower()
        if normalized_sentence not in seen_sentences:
            seen_sentences.add(normalized_sentence)
            unique_sentences.append(sentence)

    return ' '.join(unique_sentences)

# Ensure session state is initialized
if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = ""
if "pdf_tables" not in st.session_state:
    st.session_state["pdf_tables"] = []  # Initialize as an empty list

# Step 0: Upload PDF
st.sidebar.header("Upload Your Financial Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.sidebar.write(f"You uploaded: {uploaded_file.name}")

    # Extract text and tables
    pdf_text, pdf_tables = extract_text_tables_pdfplumber(uploaded_file)

    if pdf_text is not None:
        # Store results in session state
        st.session_state["pdf_text"] = pdf_text
        st.session_state["pdf_tables"] = pdf_tables  # Save tables separately

        st.sidebar.success("PDF uploaded and text extracted!")
    else:
        st.markdown("[Example Financial Documents](https://drive.google.com/drive/folders/1jMu3S7S_Hc_RgK6_cvsCqIB8x3SSS-R6)")
        st.error("No text extracted from the uploaded PDF.")

# Step 1: Display Extracted Text
st.subheader("Extracted Text")
if st.session_state["pdf_text"]:
    st.text_area("Document Text", st.session_state["pdf_text"], height=400)
else:
    st.warning("No text extracted yet. Upload a PDF to start.")


# Step 2: Display Extracted Tables (Fixed Error)
st.subheader("Extracted Tables")
if st.session_state["pdf_tables"]:  # Check if tables exist
    for idx, table in enumerate(st.session_state["pdf_tables"]):
        st.write(f"Table {idx+1}")
        st.write(pd.DataFrame(table))  # Display tables as DataFrames
else:
    st.info("No tables extracted.")

# Retrieve variables from session state
nlp = st.session_state["nlp"]
summarizer = st.session_state["summarizer"]
pdf_text = st.session_state["pdf_text"]
pdf_tables = st.session_state["pdf_tables"]

# Ensure that the models are loaded
if nlp is None or summarizer is None:
    st.error("Models are not properly loaded. Please check your model paths and installation.")
else:
    # Step 3: Named Entity Recognition (NER)
    st.subheader("NER Analysis")

    # Display full extracted text, not just first 1000 characters
    example_text = st.text_area(
        "Enter or paste text for analysis",
        height=400,
        value=st.session_state["pdf_text"] if st.session_state["pdf_text"] else ""
    )

    if st.button("Analyze"):
        # Ensure full extracted text is used for analysis
        text_for_analysis = st.session_state["pdf_text"].strip() if st.session_state["pdf_text"] else example_text.strip()
    
        if text_for_analysis:
            with st.spinner("Analyzing text..."):
                # Extract structured financial data using regex (Now using full text)
                extracted_data = {
                    key: (match.group(1) if match else "N/A")
                    for key, pattern in patterns.items()
                    if (match := re.search(pattern, text_for_analysis, re.IGNORECASE))
                }

                doc = nlp(text_for_analysis)
                financial_entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["MONEY", "PERCENT", "ORG", "DATE"]]
    
                # Store extracted data in a structured dictionary
                structured_data = {**extracted_data, "Named Entities Extracted": financial_entities}
    
                # Display results
                st.write("Entities Found:")
                st.write(pd.DataFrame(financial_entities, columns=["Entity", "Label"]))
    
                st.write("Structured Data Extracted:")
                st.write(pd.DataFrame([structured_data]))
    
        else:
            st.error("Please provide some text for analysis.")
    
    # Step 4: Summarization
    st.subheader("Summarization")
    st.write("Generate concise summaries of financial documents.")
    
    # Add customization options for summarization with chunk selection
    st.sidebar.header("Summarization Settings")
    max_chunks_to_process = st.sidebar.slider(
        "Max chunks to summarize", 
        min_value=1, 
        max_value=10, 
        value=3,
        help="Select fewer chunks for faster processing but less comprehensive summaries"
    )
    
    # Allow users to add custom keywords
    custom_keywords = st.sidebar.text_input(
        "Add custom keywords (comma separated)",
        value="",
        help="Add domain-specific keywords to improve chunk selection"
    )
    
    # Text summarization input
    input_text = st.text_area(
        "Enter text to summarize",
        height=200,
        value=st.session_state.get("pdf_text", "") if "pdf_text" in st.session_state else ""
    )
    
    # Add option to see chunk selection details
    show_chunk_details = st.sidebar.checkbox("Show chunk selection details", value=False)
    
    if st.button("Summarize"):
        text_to_summarize = input_text.strip()
        if text_to_summarize:
            try:
                # Display original text length
                input_length = len(text_to_summarize.split())
                st.write(f"Original text length: {input_length} words")
                
                # Process custom keywords if provided
                keywords = None
                if custom_keywords:
                    keywords = [kw.strip() for kw in custom_keywords.split(",") if kw.strip()]
                    st.write(f"Using custom keywords: {', '.join(keywords)}")
    
                # Split the text into manageable chunks
                chunks = split_text_into_chunks(text_to_summarize, tokenizer)
                st.write(f"Text has been split into {len(chunks)} chunks.")
                
                # NEW: Rank and select the best chunks instead of processing all of them
                selected_chunks = rank_and_select_chunks(
                    chunks, 
                    max_chunks=max_chunks_to_process,
                    keywords=keywords
                )
                
                st.write(f"Selected {len(selected_chunks)} highest-ranked chunks for summarization.")
                
                # Show chunk selection details if requested
                if show_chunk_details:
                    with st.expander("Chunk Selection Details"):
                        for i, chunk in enumerate(selected_chunks):
                            st.markdown(f"**Chunk {i+1}**")
                            st.write(f"Length: {len(chunk.split())} words")
                            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            st.write("---")
    
                # Summarize each selected chunk
                summaries = []
                with st.spinner(f"Summarizing {len(selected_chunks)} chunks..."):
                    for i, chunk in enumerate(selected_chunks):
                        st.write(f"Summarizing chunk {i+1}/{len(selected_chunks)}...")
                        # Adjust summary length parameters as needed
                        chunk_length = len(chunk.split())
                        max_summary_length = min(150, chunk_length // 2)
                        min_summary_length = max(50, max_summary_length // 2)
        
                        try:
                            summary_output = summarizer(
                                chunk,
                                max_length=max_summary_length,
                                min_length=min_summary_length,
                                do_sample=False,
                                truncation=True
                            )
                            chunk_summary = summary_output[0]['summary_text'].strip()
        
                            if not chunk_summary:
                                st.warning(f"The summary for chunk {i+1} is empty.")
                            else:
                                summaries.append(chunk_summary)
        
                        except Exception as e:
                            st.error(f"Summarization failed for chunk {i+1}: {e}")
                            st.text(traceback.format_exc())
                            continue
    
                if summaries:
                    # Combine summaries and remove duplicates
                    combined_summary = ' '.join(summaries)
                    final_summary = remove_duplicate_sentences(combined_summary)
                    
                    # Calculate compression ratio
                    summary_length = len(final_summary.split())
                    compression_ratio = (1 - summary_length / input_length) * 100
                    
                    st.subheader("Final Summary")
                    st.success(final_summary)
                    st.write(f"Summary length: {summary_length} words ({compression_ratio:.1f}% compression)")
                    
                    # Display summary statistics
                    st.subheader("Summary Statistics")
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        st.metric("Original Length", f"{input_length} words")
                        st.metric("Total Chunks", str(len(chunks)))
                    with stats_col2:
                        st.metric("Summary Length", f"{summary_length} words")
                        st.metric("Chunks Processed", str(len(selected_chunks)))
                    
                else:
                    st.error("No summaries were generated.")
    
            except Exception as e:
                st.error("An error occurred during summarization.")
                st.text(traceback.format_exc())
        else:
            st.error("Please provide text to summarize.")
            
    # Add help information
    st.sidebar.markdown("---")
    with st.sidebar.expander("How Chunk Selection Works"):
        st.markdown("""
        The chunk selection algorithm ranks text chunks based on:
        
        1. **Keyword density** - Presence of financial terms
        2. **Length** - Longer chunks often contain more information
        3. **Numbers** - Financial documents with numbers are often important
        4. **Structure** - Lists and bullet points signal key information
        5. **Headers** - Section headers often introduce important content
        
        Adjust the settings above to customize the selection process.
        """)