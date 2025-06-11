# backend/ingest.py

import fitz  # PyMuPDF is needed to open the PDF and render pages for Tesseract to work on it
from PIL import Image
import pytesseract
import io
import streamlit as st
from datetime import datetime
import os

def extract_metadata_and_text(path: str) -> tuple[list[dict], dict]:
    """ This takes the PDF and extracts the metadata as well as the text and returns a tuple which has the list of dictionaries containing the metadata such as title 
    among other things"""
    page_data = []
    doc_metadata = {
        "author": "Unknown",
        "creation_date": None,
        "title": "Unknown"
    }
    doc_name = os.path.basename(path)

    try:
        doc = fitz.open(path)
        
        # Extract Document-Level Metadata such as author, date and title
        meta = doc.metadata
        doc_metadata["author"] = meta.get('author', 'Unknown').strip() or "Unknown"
        doc_metadata["title"] = meta.get('title', 'Unknown').strip() or "Unknown"
        
        # Parse creation date
        creation_date_str = meta.get('creationDate', '')
        if creation_date_str and creation_date_str.startswith('D:'):
            try:
                date_str = creation_date_str[2:16]
                doc_metadata["creation_date"] = datetime.strptime(date_str, '%Y%m%d%H%M%S')
            except (ValueError, IndexError):
                doc_metadata["creation_date"] = None
        
        st.toast(f"Starting OCR for all pages of '{doc_name}'...")

        # Applying OCR on the pages.
        for page_num, page in enumerate(doc):
            text = "" # Takes the text from each page and then for every page, resets it
            try:
                # Render the page as a high-resolution image
                pix = page.get_pixmap(dpi=300) 
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                # Use Tesseract to extract text from the image
                text = pytesseract.image_to_string(img)
            except Exception as ocr_e:
                # Toast to show unsuccessful OCR try on the page and continue
                st.toast(f"⚠️ OCR failed on page {page_num + 1}: {ocr_e}")
                text = ""
            
            # Add the extracted text to our data if any was found
            if text.strip():
                page_data.append({
                    "text": text.strip(),
                    "page_number": page_num + 1
                })
                
        doc.close()
        st.toast(f"✅ OCR complete. Extracted text from {len(page_data)} pages of '{doc_name}'.")
        
    except Exception as e:
        st.toast(f"❌ Error processing PDF '{doc_name}': {e}")
        return [], {}

    return page_data, doc_metadata
