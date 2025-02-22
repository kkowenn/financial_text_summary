import streamlit as st
import fitz  # PyMuPDF
import spacy
from transformers import pipeline
import pandas as pd

# Set the title and layout
st.set_page_config(page_title="FinBrief: Financial Document Insights", layout="wide")
st.title("FinBrief: Financial Document Insights")
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to:", ["Upload PDF", "Analysis", "Summarization"])

# Custom styling
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

# Step 1: Upload PDF
if page == "Upload PDF":
    st.subheader("Upload Your Financial Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.write(f"You uploaded: {uploaded_file.name}")

        # Extract text from the uploaded PDF
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text()

        if pdf_text:
            st.write("Extracted Text:")
            st.text_area("Document Text", pdf_text[:1000], height=300)  # Display first 1000 characters
        else:
            st.error("No text found in the uploaded PDF.")

# Step 2: Named Entity Recognition (NER)
elif page == "Analysis":
    st.subheader("NER Analysis")
    st.write("This section identifies key entities like monetary values, percentages, and dates.")

    # Load the spaCy model once at the start
    try:
        nlp = spacy.load("en_core_web_sm")  # Ensure spaCy's model is installed
    except OSError:
        nlp = None  # Handle the case where the model is missing

    # Example NER with sample text
    example_text = st.text_area("Enter or paste text for analysis", height=200)

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Analyze"):
            if example_text:
                with st.spinner("Analyzing text..."):
                    doc = nlp(example_text)
                    entities = [(ent.text, ent.label_) for ent in doc.ents]
                    st.write("Entities Found:")
                    st.write(pd.DataFrame(entities, columns=["Entity", "Label"]))
            else:
                st.error("Please provide some text for analysis.")

    with col2:
        st.markdown(
            """
            **Tip:** You can paste a section of your financial document to detect key metrics like revenue, expenses, and dates.
            """
        )

# Step 3: Summarization
elif page == "Summarization":
    st.subheader("Summarization")
    st.write("Generate concise summaries of financial documents.")

    # Load the custom financial summarization model
    try:
        max_input_length = 1024
        summarizer = pipeline("summarization", model="kritsadaK/bart-financial-summarization")
    except Exception as e:
        st.error("Summarization model not loaded. Ensure PyTorch/TensorFlow is installed.")
        st.write(e)

    # Text summarization
    input_text = st.text_area("Enter text to summarize", height=200)

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Summarize"):
            if input_text:
                if len(input_text.split()) > max_input_length:
                    st.error(f"Input text is too long. Please provide fewer than {max_input_length} words.")
                else:
                    with st.spinner("Generating summary..."):
                        summary = summarizer(input_text, max_length=256, min_length=50, do_sample=False)
                        st.write("Summary:")
                        st.success(summary[0]["summary_text"])
            else:
                st.error("Please provide text to summarize.")

    with col2:
        st.markdown(
            """
            **Tip:** Keep your input concise. If the document is long, break it into smaller sections.
            """
        )
