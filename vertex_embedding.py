# vertex_embedding.py

import os
import streamlit as st
from langchain_google_vertexai import VertexAIEmbeddings

class EmbeddingClient:
    """
    Vertex AI Embeddings：
      - embed_query(text)
      - embed_documents([text1, text2, ...])
    """
    def __init__(self, model_name, location):
        self.client = VertexAIEmbeddings(
            model_name=model_name,
            location=location
        )

    def embed_query(self, query):
        try:
            return self.client.embed_query(query)
        except Exception as e:
            st.error(f"Error embedding query: {e}")
            return None

    def embed_documents(self, documents):
        try:
            return self.client.embed_documents(documents)
        except Exception as e:
            st.error(f"Error embedding documents: {e}")
            return None
