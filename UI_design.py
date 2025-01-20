import sys
import os
import streamlit as st

# Assuming these four files are in the same folder
sys.path.append(os.path.abspath('.'))

from File_uploader import DocumentProcessor
from vertex_embedding import EmbeddingClient
from integration import ChromaCollectionCreator

def main():
    st.title("Chroma Collection Manager - Multi-chunk, Multi-result Demo")

    # ====== 1) Set your Google Cloud credentials and Vertex AI Embeddings configuration ======
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/liuhaibo/gcloud_key/geminisample-425519-a9f2d1d62e3b.json"  # Replace with the actual path
    embed_config = {
        "model_name": 'textembedding-gecko@003',
        "location": 'us-central1'
    }

    # ====== 2) Initialize DocumentProcessor (Upload PDFs) ======
    processor = DocumentProcessor()
    processor.ingest_documents()

    # ====== 3) Initialize EmbeddingClient (Vertex AI) ======
    embed_client = EmbeddingClient(**embed_config)

    # ====== 4) Initialize ChromaCollectionCreator ======
    chroma_creator = ChromaCollectionCreator(processor, embed_client)

    # ====== 5) Form: Click the button to create Chroma, input a query topic & specify the number of results ======
    with st.form("create_and_search_chroma"):
        st.subheader("Build & Query Chroma DB")
        topic_query = st.text_input("Enter a query to search the Chroma Collection:")
        num_questions = st.slider("Number of Results (k)", 1, 10, 3)

        submitted = st.form_submit_button("Create & Search!")
        if submitted:
            # a) Create/refresh Chroma
            chroma_creator.create_chroma_collection()

            # b) Query
            results = chroma_creator.query_chroma_collection(topic_query, k=num_questions)
            if results:
                st.write(f"**Top {num_questions} results:**")
                for i, (doc, score) in enumerate(results, start=1):
                    st.subheader(f"Result {i}")
                    # Display the Python representation of the Document
                    st.code(f"{repr(doc)}", language="python")
                    st.write("**Similarity Score:**", score)
                    st.write("---")

if __name__ == '__main__':
    main()
