# Quizzify: AI-Powered Quiz Generator

Quizzify is a Streamlit-based web application designed to generate intelligent quizzes from uploaded PDFs using AI-powered embeddings and contextual information. It leverages Google Vertex AI, LangChain libraries, and a custom Chroma vector database to create dynamic, topic-specific quizzes.

---

## Demo
[Watch the Demo](https://youtu.be/W7-s1UqQsFA)

## Features

- **PDF Upload and Processing**: 
  Upload multiple PDF files and process them into manageable chunks with metadata (source, page number, and unique identifier).

- **Vector Store Integration**:
  Store processed document embeddings in a Chroma vector database for efficient similarity-based retrieval.

- **AI-Generated Quizzes**:
  Dynamically generate multiple-choice quizzes based on user-defined topics and context from the uploaded documents.

- **Interactive Quiz Management**:
  Navigate between questions, view explanations, and interact with the quiz through a user-friendly interface.

---

## How It Works

1. **Upload PDFs**:
   - Users upload one or more PDF documents.
   - The application processes the documents into pages and splits them into chunks for embedding.

2. **Embedding and Vector Storage**:
   - Document chunks are embedded using Google Vertex AI.
   - The embeddings are stored in a Chroma vector store for later retrieval.

3. **Quiz Generation**:
   - Users provide a topic for the quiz.
   - The application uses LangChain’s `PromptTemplate` and Vertex AI to generate multiple-choice questions.

4. **Quiz Interaction**:
   - Users answer questions, view explanations, and navigate between questions.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Tingz-sky/Gemini_Quizzify.git
   cd quizzify
   ```
   
2. **Install Dependencies: Ensure Python 3.8 or higher is installed. Run**:
  ```bash
  pip install -r requirements.txt
  ```

3. **Set Up Google Cloud Credentials**:
- Obtain a service account JSON key for Google Cloud.
- Set the GOOGLE_APPLICATION_CREDENTIALS environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"
```
4. **Run the Application**:
```bash
streamlit run quizzify.py
```

## File Structure
- **File_uploader.py**: Handles PDF uploads and splits them into manageable chunks with metadata.

- **vertex_embedding.py**: Contains the EmbeddingClient class for embedding text using Google Vertex AI.

- **integration.py**: Manages the storage and retrieval of document embeddings using Chroma.

- **quiz_algo.py**: Implements the QuizGenerator class for creating quizzes based on topics and context.

- **generate_quiz.py**: Provides an entry point for quiz generation and testing.

- **UI_design.py**: Offers a user-friendly interface for database queries and quiz interaction.

- **ui.py**: Integrates various modules for an interactive quiz manager.

- **quizzify.py**: Main entry point for the application that combines all functionalities.

## Requirements
- Python 3.8 or higher
- Google Vertex AI access
- Libraries:
    - Streamlit
    - LangChain
    - PyPDFLoader
    - Chroma
## Authors
Developed by Tianze Zhang. For questions or support, contact tianze.zhang@mail.utoronto.ca.
