import streamlit as st
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
import os
import sys

sys.path.append(os.path.abspath('../../'))


class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        Initializes the QuizGenerator with a required topic, the number of questions for the quiz,
        and an optional vectorstore for querying related information.
        """
        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.system_template = """
            You are a subject matter expert on the topic: {topic}

            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"

            You must respond as a JSON object with the following structure:
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}

            Context: {context}
        """

    def init_llm(self):
        """
        Initializes the Large Language Model (LLM) for quiz question generation.
        """
        self.llm = VertexAI(
            model_name="gemini-pro",
            temperature=0.7,
            max_output_tokens=512,
            top_p=0.8,
            top_k=40
        )

    def generate_question_with_vectorstore(self):
        """
        Generates a quiz question using the provided topic and context from the vectorstore.
        """
        # 1) Ensure both LLM and vectorstore are initialized
        if self.llm is None:
            self.init_llm()  # Initialize if not already done
        if self.vectorstore is None:
            raise ValueError("Vectorstore not provided. Provide a vectorstore to use context.")

        from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        # 2) Get the retriever
        #   Use self.vectorstore.db.as_retriever(...)
        retriever = self.vectorstore.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.num_questions}  # or k=3
        )

        # 3) Create a PromptTemplate
        #   Use PromptTemplate.from_template(...) or PromptTemplate(template=..., input_variables=[...])
        prompt = PromptTemplate.from_template(self.system_template)

        # 4) Build the Runnable for parallel execution: retrieve context + topic
        setup_and_retrieval = RunnableParallel(
            {
                "context": retriever,  # Use retriever.invoke(...) to get documents
                "topic": RunnablePassthrough()  # Pass self.topic directly to the next step
            }
        )

        # 5) Fill the prompt with the retrieved results + topic
        #   Finally, chain it with the LLM
        chain = setup_and_retrieval | prompt | self.llm

        # 6) Execute the chain
        response = chain.invoke(self.topic)
        return response


# Test the Object
if __name__ == "__main__":
    from File_uploader import DocumentProcessor
    from vertex_embedding import EmbeddingClient
    from integration import ChromaCollectionCreator

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/liuhaibo/gcloud_key/geminisample-425519-a9f2d1d62e3b.json"
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "location": "us-central1"
    }

    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)  # Initialize from Task 4
        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        question = None

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for ingestion, specify the topic for the quiz, and click Generate!")

            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)

            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                st.write(topic_input)

                # Test the Quiz Generator
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question = generator.generate_question_with_vectorstore()

    if question:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            st.write(question)
