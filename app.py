import streamlit as st
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
import PyPDF2
#from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

# Load sentence embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Split abstracts using the "Abstract Number:" marker
def split_abstracts_by_marker(text, marker="Abstract Number:"):
    parts = text.split(marker)
    abstracts = []
    for part in parts[1:]:
        clean = marker + part.strip()
        abstracts.append(clean)
    return abstracts

# Build FAISS index
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Semantic search
def semantic_search(query, model, abstracts, index, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(I[0]):
        results.append({
            "abstract": abstracts[idx],
            "score": float(D[0][rank])
        })
    return results


def highlight_semantic_keywords(text, query, model, threshold=0.7):
    # Tokenize and filter words
    text_words = re.findall(r'\b\w+\b', text)
    query_words = re.findall(r'\b\w+\b', query.lower())

    # Remove short or stop-like words
    text_words_filtered = [w for w in text_words if len(w) > 2]
    query_words_filtered = [w for w in query_words if len(w) > 2]

    # Embed query and text words
    query_vecs = model.encode(query_words_filtered)
    text_vecs = model.encode(text_words_filtered)

    # Compute cosine similarity between each text word and all query words
    sim_matrix = cosine_similarity(text_vecs, query_vecs)
    max_similarities = np.max(sim_matrix, axis=1)

    # Collect words that exceed similarity threshold
    highlight_words = {
        word for word, score in zip(text_words_filtered, max_similarities) if score >= threshold
    }

    # Highlight matching words in original text
    def replacer(match):
        word = match.group(0)
        if word in highlight_words:
            return f'<mark style="background-color: #ffe599;">{word}</mark>'
        return word

    return re.sub(r'\b\w+\b', replacer, text)

# Streamlit layout
st.set_page_config(page_title="PDF Semantic Search", layout="wide")
st.title("📄🔍 Semantic Search on PDF Abstracts")

model = load_model()
uploaded_file = st.file_uploader("Upload a PDF containing abstracts", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and processing abstracts..."):
        text = extract_text_from_pdf(uploaded_file)
        abstracts = split_abstracts_by_marker(text)

        if len(abstracts) == 0:
            st.error("No abstracts found. Make sure 'Abstract Number:' appears in the PDF.")
        else:
            embeddings = model.encode(abstracts, convert_to_numpy=True)
            index = build_faiss_index(embeddings)
            st.success(f"✅ Loaded {len(abstracts)} abstracts.")

            query = st.text_input("Enter your semantic search query:", placeholder="e.g., immunotherapy in melanoma")

            if query:
                top_k = st.slider("Number of top results", 1, min(10, len(abstracts)), value=3)

                with st.spinner("Running semantic search..."):
                    results = semantic_search(query, model, abstracts, index, top_k)

                st.subheader("🔎 Top Results")
                for i, res in enumerate(results):
                 #   highlighted = highlight_keywords_fuzzy(res["abstract"], query)
                    highlighted = highlight_semantic_keywords(res["abstract"], query, model)
                    st.markdown(f"### {i+1}. Score: `{res['score']:.4f}`")
                    st.markdown(highlighted, unsafe_allow_html=True)
                    st.markdown("---")
else:
    st.info("Please upload a PDF to begin.")

