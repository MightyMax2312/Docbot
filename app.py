import streamlit as st
from datetime import datetime
import tempfile
import os

# Import all the py files that serve as the backend
from backend.ingest import extract_metadata_and_text
from backend.vectorstore import create_faiss_index, search_index
from backend.qa import ask_groq 
# Import all the necessary functions required

 
st.set_page_config(
    page_title="Doc Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for the query bar 
st.markdown("""
<style>
    /* Add padding to the bottom of the main content area to make space for the fixed bar */
    section[data-testid="st.main"] > div:first-child {
        padding-bottom: 120px; 
    }

    /* The outer container for the input bar, fixed to the bottom of the viewport */
    .fixed-input-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        padding: 1rem 1rem; /* Padding */
        background-color: #0e1117; /* Color */
        border-top: 1px solid #262730;
        z-index: 999;
    }
    
    .input-bar-content {
        max-width: 730px; 
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)


# Initialising all the necessary state variables
def initialize_session_state():
    defaults = {
        "active_chat": None,
        "chats": {},
        "chat_docs": {},
        "chat_indexes": {},
        "doc_details": {},
        "editing_index": None,
        "viewing_doc_name": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()


# Making the sidebar for Chat and Document Management using streamlit
def display_sidebar():
    """Renders the sidebar for chat management and filtering."""
    with st.sidebar:
        st.title("💬 Chat & Filters")

        if st.button("➕ New Chat", use_container_width=True):
            chat_name = f"Chat {len(st.session_state.chats) + 1}"
            st.session_state.active_chat = chat_name
            st.session_state.chats[chat_name] = []
            st.session_state.editing_index = None
            st.rerun()

        st.subheader("Your Chats")
        for name in list(st.session_state.chats.keys()):
            col1, col2 = st.columns([0.85, 0.15])
            button_type = "primary" if name == st.session_state.active_chat else "secondary"
            if col1.button(name, key=f"chat_{name}", use_container_width=True, type=button_type):
                st.session_state.active_chat = name
                st.session_state.editing_index = None
                st.rerun()
            if col2.button("🗑️", key=f"del_{name}", help=f"Delete chat '{name}'"):
                delete_chat(name)
                st.rerun()
        
        st.divider()

        active_chat = st.session_state.active_chat
        if active_chat:
            display_document_controls(active_chat)

# Renders UI for file filtering and manual re-indexing.
def display_document_controls(chat_name):
    st.subheader("Documents & Search Filters")
    
    chat_docs = st.session_state.chat_docs.get(chat_name, [])
    if not chat_docs:
        st.info("Upload documents using the ➕ icon in the chat input to start.")
        return

    if st.button("🔄 Re-Index All Documents", use_container_width=True):
        handle_indexing(chat_name)

    # Simplified Filtering UI for document searching
    st.markdown("##### Filter Your Search")
    included_docs = st.multiselect("Include documents:", options=chat_docs, default=chat_docs, key=f"include_{chat_name}")

    # Store the remaining filter in session state
    st.session_state[f'filters_{chat_name}'] = {'included_docs': included_docs}

    # Section to view full document text due to OCR
    st.divider()
    st.markdown("##### View Documents")
    for doc_name in chat_docs:
        if st.button(doc_name, key=f"view_{doc_name}", use_container_width=True):
            st.session_state.viewing_doc_name = doc_name
            st.rerun()


# UI: Main Chat Interface 
def display_main_content():
    st.header("Doc Chat")

    # The document viewer is now an expander at the top of the content area
    if st.session_state.get("viewing_doc_name"):
        render_document_viewer_expander()

    active_chat = st.session_state.active_chat
    if not active_chat:
        st.info("Create a new chat or select one from the sidebar to begin.")
        return

    st.subheader(f"Conversation: {active_chat}")

    messages = st.session_state.chats.get(active_chat, [])
    for i, msg in enumerate(messages):
        if st.session_state.editing_index == i:
            render_edit_form(i, msg, active_chat)
        else:
            render_message(i, msg)
    
    st.markdown('<div class="fixed-input-bar"><div class="input-bar-content">', unsafe_allow_html=True)
    if st.session_state.editing_index is None:
        render_chat_input_bar(active_chat)
    st.markdown('</div></div>', unsafe_allow_html=True)


def render_document_viewer_expander():
    # Expands the document to view the contents
    doc_name = st.session_state.viewing_doc_name
    if doc_name not in st.session_state.doc_details:
        st.error(f"Could not find details for document: {doc_name}")
        st.session_state.viewing_doc_name = None
        return
    with st.expander(f"📄 Viewing: {doc_name}", expanded=True):
        full_text = "\n\n".join(
            page['text'] for page in st.session_state.doc_details[doc_name].get('pages', [])
        )
        st.text_area("Document Content", value=full_text, height=500, disabled=True, label_visibility="collapsed")
        if st.button("Close Viewer", key="close_doc_viewer"):
            st.session_state.viewing_doc_name = None
            st.rerun()

def render_edit_form(index, msg, chat_name):
    # Query editing
    with st.form(key=f"edit_form_{index}"):
        edited_prompt = st.text_area("Edit prompt:", value=msg["content"], label_visibility="collapsed")
        c1, c2 = st.columns([1, 1])
        if c1.form_submit_button("Save & Resubmit", use_container_width=True, type="primary"):
            st.session_state.editing_index = None
            handle_user_query(edited_prompt, chat_name, is_edit=True, edit_index=index)
        if c2.form_submit_button("Cancel", use_container_width=True):
            st.session_state.editing_index = None
            st.rerun()

def render_message(index, msg):
    with st.chat_message(msg["role"]):
        col1, col2 = st.columns([0.95, 0.05])
        with col1:
            st.markdown(msg["content"])
        if msg["role"] == "user":
            with col2:
                if st.button("✏️", key=f"edit_btn_{index}", help="Edit this prompt"):
                    st.session_state.editing_index = index
                    st.rerun()
        if msg["role"] == "assistant" and "chunks" in msg:
            display_citations(msg.get("chunks", []))

def render_chat_input_bar(chat_name):
    # Uploading PDFs option
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        with st.popover("➕", use_container_width=False):
            st.markdown("Attach and index new documents:")
            st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key=f"popover_uploader_{chat_name}", label_visibility="collapsed", on_change=process_and_index_uploads)
    with col2:
        prompt = st.chat_input("Ask a question...")
        if prompt:
            handle_user_query(prompt, chat_name)

def process_and_index_uploads():
    # Indexing the document after uploading
    chat_name = st.session_state.active_chat
    if not chat_name: return
    uploader_key = f"popover_uploader_{chat_name}"
    if uploader_key in st.session_state and st.session_state[uploader_key]:
        uploaded_files = st.session_state[uploader_key]
        handle_file_upload(uploaded_files, chat_name)
        handle_indexing(chat_name)

def handle_file_upload(uploaded_files, chat_name):
    with st.spinner("Processing uploaded files..."):
        if chat_name not in st.session_state.chat_docs:
            st.session_state.chat_docs[chat_name] = []
        for file in uploaded_files:
            if file.name not in st.session_state.doc_details:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    if tmp_path:
                        pages, metadata = extract_metadata_and_text(tmp_path)
                        if pages:
                            st.session_state.doc_details[file.name] = {'pages': pages, 'metadata': metadata}
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
            if file.name not in st.session_state.chat_docs[chat_name]:
                st.session_state.chat_docs[chat_name].append(file.name)
    st.toast(f"{len(uploaded_files)} file(s) processed.")

def handle_indexing(chat_name):
    with st.spinner("Indexing documents... This may take a moment."):
        docs_to_index = {name: st.session_state.doc_details[name] for name in st.session_state.chat_docs.get(chat_name, []) if name in st.session_state.doc_details}
        if not docs_to_index:
            st.warning("No documents available to index.")
            return
        index, metadata = create_faiss_index(docs_to_index)
        if index is not None:
            st.session_state.chat_indexes[chat_name] = {'index': index, 'metadata': metadata}
            st.toast("Documents indexed successfully!")

def handle_user_query(user_input, chat_name, is_edit=False, edit_index=None):
    if is_edit and edit_index is not None:
        st.session_state.chats[chat_name] = st.session_state.chats[chat_name][:edit_index]
    st.session_state.chats[chat_name].append({"role": "user", "content": user_input})
    with st.spinner("Searching and generating answer..."):
        response, top_chunks = "", []
        index_data = st.session_state.chat_indexes.get(chat_name)
        if index_data:
            filters = st.session_state.get(f'filters_{chat_name}', {})
            initial_results = search_index(user_input, index_data["index"], index_data["metadata"], top_k=25)
            filtered_results = apply_search_filters(initial_results, filters)
            top_chunks = filtered_results[:5]
        if top_chunks:
            st.toast("✅ Found relevant context in your documents.")
            context = build_context_from_chunks(top_chunks)
            response = ask_groq(user_input, context)
        else:
            if index_data: st.toast("ℹ️ No specific context found. Answering generally.")
            response = ask_groq(user_input, context=None) 
        st.session_state.chats[chat_name].append({"role": "assistant", "content": response, "chunks": top_chunks})
    st.rerun()

def apply_search_filters(results, filters):
    # Manually select and deselect documents to search from them
    if not filters or 'included_docs' not in filters:
        return results
    
    included_docs = filters['included_docs']
    if not included_docs:
        return results 
    """ This filter first gets all the context and then depending on the deselected documents, it gets rid of the contexts that isn't included. """
    return [chunk for chunk in results if chunk['doc_name'] in included_docs]

def build_context_from_chunks(chunks):
    if not chunks: return None
    return "\n\n---\n\n".join(
        f"Source [{i+1}] from Document '{chunk['doc_name']}', Page {chunk['page_number']}:\nContent: {chunk['text']}"
        for i, chunk in enumerate(chunks)
    )

def display_citations(chunks):
    # For displaying the particular contexts and the page number along with the phrase
    if not chunks: return
    with st.expander("📄 View Retrieved Sources"):
        for i, chunk in enumerate(chunks):
            citation_title = f"**Source [{i+1}]**: {chunk['doc_name']} (Page {chunk['page_number']})"
            st.markdown(f"##### {citation_title}")
            st.markdown(f"> {chunk['text']}")
            if i < len(chunks) - 1: st.divider()

def delete_chat(chat_name):
    # The option to delete the chats
    st.session_state.chats.pop(chat_name, None)
    st.session_state.chat_docs.pop(chat_name, None)
    st.session_state.chat_indexes.pop(chat_name, None)
    st.session_state.pop(f'filters_{chat_name}', None)
    if st.session_state.active_chat == chat_name:
        st.session_state.active_chat = None
        st.session_state.editing_index = None

# Main app
if __name__ == "__main__":
    display_sidebar()
    display_main_content()
