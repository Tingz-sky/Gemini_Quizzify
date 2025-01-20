import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# Import the custom EmbeddingClient.
# Note: The main_app.py file will import this file, and it also imports vertex_embedding.
# So, the object will be injected later.
# If you encounter errors, you can change this line to: `from vertex_embedding import EmbeddingClient`.
# However, be mindful of circular import issues.

class VertexEmbeddings(Embeddings):
    """
    Adapts the langchain Embeddings interface, allowing Chroma to directly use our Vertex EmbeddingClient.
    """
    def __init__(self, embed_client):
        self.embed_client = embed_client  # This is an instance of EmbeddingClient.

    def embed_documents(self, texts):
        return self.embed_client.embed_documents(texts)

    def embed_query(self, query):
        return self.embed_client.embed_query(query)


class ChromaCollectionCreator:
    """
    Responsible for splitting PDF documents into small chunks, storing them in Chroma, and supporting multiple similarity searches.
    """
    def __init__(self, processor, embed_model):
        """
        processor: Instance of DocumentProcessor (contains all pages of the PDF).
        embed_model: Instance of EmbeddingClient (Vertex AI).
        """
        self.processor = processor
        self.embed_model = embed_model
        self.db = None  # Stores the Chroma vector store.

    def create_chroma_collection(self):
        """
        1. Access processor.pages, where each page is a Document (page_content, metadata).
        2. Split the content into chunks of a specified size and store the chunks in Chroma.
        """
        if len(self.processor.pages) == 0:
            st.error("No documents found!")
            return

        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        doc_list = []

        # Optional: Perform deduplication to avoid duplicate chunks (if deemed useful).
        seen_texts = set()

        for page_doc in self.processor.pages:
            # page_doc is a Document with page_content and metadata.
            chunks = splitter.split_text(page_doc.page_content)
            for idx, chunk_text in enumerate(chunks, start=1):
                # Copy the original metadata.
                new_meta = dict(page_doc.metadata)
                new_meta["chunk_index"] = idx

                # Deduplication logic (optional).
                # If the chunk_text is large, you can use hash(chunk_text) or a more complex text similarity method.
                text_hash = hash(chunk_text.strip())
                if text_hash in seen_texts:
                    continue
                seen_texts.add(text_hash)

                doc_list.append(Document(page_content=chunk_text, metadata=new_meta))

        st.success(f"Successfully split pages into {len(doc_list)} text chunks!")

        # Wrap EmbeddingClient using VertexEmbeddings.
        embedding = VertexEmbeddings(self.embed_model)

        try:
            # Use from_documents() to store chunks.
            self.db = Chroma.from_documents(
                documents=doc_list,
                embedding=embedding,
                persist_directory="./chroma_db"
            )
            st.success("Successfully created Chroma Collection!")
        except Exception as e:
            st.error(f"Failed to create Chroma Collection: {e}")

    def query_chroma_collection(self, query, k=1):
        """
        Returns the top-k matching results: [(Document, score), ...].
        """
        if not self.db:
            st.error("Chroma Collection has not been created!")
            return None

        docs = self.db.similarity_search_with_relevance_scores(query, k=k)
        if docs:
            return docs
        else:
            st.error("No matching documents found!")
            return None
