import os
import uuid
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

class DocumentProcessor:
    def __init__(self):
        self.pages = []  # Store all Documents, where each Document corresponds to a page of a PDF

    def ingest_documents(self):
        """
        Allow users to upload multiple PDFs and process them page by page. Each page is assigned basic metadata:
        - source: the file name
        - page:   the current page number
        - pdf_uuid: a unique identifier generated for the entire PDF
        """
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Generate a globally unique identifier for this PDF
                pdf_uuid = str(uuid.uuid4())

                # Save the PDF as a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file_name = temp_file.name
                    temp_file.write(uploaded_file.getvalue())

                try:
                    # Use PyPDFLoader to read the PDF page by page
                    loader = PyPDFLoader(temp_file_name)
                    extracted_pages = loader.load()  # List[Document]

                    # Add metadata to each Document page
                    for idx, page in enumerate(extracted_pages, start=1):
                        page.metadata["source"] = uploaded_file.name
                        page.metadata["page"] = idx
                        page.metadata["pdf_uuid"] = pdf_uuid

                    # Append to the overall pages list
                    self.pages.extend(extracted_pages)

                finally:
                    # Delete the temporary PDF file
                    os.unlink(temp_file_name)

            st.write(f"Total pages processed: {len(self.pages)}")
