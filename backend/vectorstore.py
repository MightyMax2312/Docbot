from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

# The embedder is responsible for converting the text to vectors which can be used for comparison for further use
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Failed to load SentenceTransformer model: {e}. Please check your internet connection and dependencies.")
    st.stop()

# Initialize a robust text splitter from LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,      # Increased chunk size for more context
    chunk_overlap=150,   # Overlap in order to ensure that the other chunks has more context to work with
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""] # Order of separators for splitting
)


def create_faiss_index(docs_with_meta: dict):
    # Creates a FAISS index from a dictionary of documents and their metadata. This receives the dictionary and then returns the Faiss index and the metadata chunks
    all_chunks_text = []
    all_chunks_metadata = []
    global_chunk_id = 0

    if not docs_with_meta:
        st.warning("No documents provided to create FAISS index.")
        return None, []

    for doc_name, doc_data in docs_with_meta.items():
        pages_data_list = doc_data.get('pages', [])
        doc_meta = doc_data.get('metadata', {}) # Author, creation date, etc.

        if not pages_data_list:
            continue

        for page_entry in pages_data_list:
            page_text = page_entry.get('text', '')
            page_number = page_entry.get('page_number')

            # Use the robust text splitter
            chunks_from_page = text_splitter.split_text(page_text)

            for chunk_content in chunks_from_page:
                if len(chunk_content.strip()) > 25: # Ensure chunks are meaningful
                    all_chunks_text.append(chunk_content)
                    
                    # Metadata for each chunk
                    chunk_meta = {
                        "doc_name": doc_name,
                        "chunk_id": f"{doc_name}_page{page_number}_chunk{global_chunk_id}", # Unique ID for citations
                        "text": chunk_content,
                        "page_number": page_number,
                    }
                    # Add the document-level metadata to the chunk-level metadata
                    chunk_meta.update(doc_meta)
                    all_chunks_metadata.append(chunk_meta)
                    
                    global_chunk_id += 1

    if not all_chunks_text:
        st.toast("❌ No valid text chunks were found to index.")
        return None, []

    # Embed and Create FAISS Index
    try:
        with st.spinner(f"Generating {len(all_chunks_text)} embeddings..."):
            vectors = embedder.encode(all_chunks_text, show_progress_bar=True)
        
        vectors = np.array(vectors, dtype='float32')
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        st.toast(f"✅ FAISS index created with {index.ntotal} vectors.")
        return index, all_chunks_metadata
        
    except Exception as e:
        st.error(f"Error during embedding or indexing: {e}")
        return None, []


def search_index(query: str, index: faiss.Index, metadata: list[dict], top_k: int = 10) -> list[dict]:
    # Searches the FAISS index and returns the top matching chunks using vector distancing along with the full metadata.

    if index is None:
        return []

    try:
        query_vector = embedder.encode([query], convert_to_tensor=False).astype('float32')
        distances, indices = index.search(query_vector, top_k)
        
        results = [metadata[i] for i in indices[0] if 0 <= i < len(metadata)]
        return results
        
    except Exception as e:
        st.error(f"❌ Error during FAISS search: {e}")
        return []

