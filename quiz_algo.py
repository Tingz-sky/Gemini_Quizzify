import streamlit as st
import os
import sys
import json

sys.path.append(os.path.abspath('../../'))
from File_uploader import DocumentProcessor
from vertex_embedding import EmbeddingClient
from integration import ChromaCollectionCreator

from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI


class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        Initializes the QuizGenerator with a required topic, the number of questions for the quiz,
        and an optional vectorstore for querying related information.

        :param topic: A string representing the required topic of the quiz.
        :param num_questions: An integer representing the number of questions to generate for the quiz, up to a maximum of 10.
        :param vectorstore: An optional vectorstore instance (e.g., ChromaDB) to be used for querying information related to the quiz topic.
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
        self.question_bank = []  # Initialize the question bank to store questions

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
        Initializes and configures the Large Language Model (LLM) for generating quiz questions.
        """
        self.llm = VertexAI(
            model_name="gemini-pro",  # Or chat-gemini@001, text-bison@002, etc.
            temperature=0.8,  # A bit higher for variability
            max_output_tokens=500
        )

    def generate_question_with_vectorstore(self):
        """
        Generates a quiz question based on the topic provided using a vectorstore.

        :return: A JSON string representing the generated quiz question (or some text that
                 should ideally be in JSON format).
        """
        if not self.llm:
            self.init_llm()
        if not self.vectorstore:
            raise ValueError("Vectorstore not provided.")

        from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        # 1) Enable a Retriever
        # If your vectorstore is a ChromaCollectionCreator, it may expose .db, so check carefully.
        # If it directly exposes as_retriever, we can just do:
        retriever = self.vectorstore.db.as_retriever()

        # 2) Use the system template to create a PromptTemplate
        prompt = PromptTemplate.from_template(self.system_template)

        # 3) RunnableParallel: get {context, topic} from retriever + passthrough
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )

        # 4) Create a chain: retrieve -> prompt -> LLM
        chain = setup_and_retrieval | prompt | self.llm

        # 5) Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response

    def generate_quiz(self) -> list:
        """
        Task: Generate a list of unique quiz questions based on the specified topic and number of questions.
        """
        self.question_bank = []  # Reset the question bank

        for _ in range(self.num_questions):
            # 1. Use class method to generate question (JSON string).
            question_str = self.generate_question_with_vectorstore()
            # 2. Try to parse the JSON
            try:
                question_dict = json.loads(question_str)
            except json.JSONDecodeError:
                print("Failed to decode question JSON.")
                continue  # Skip if JSON decoding fails

            # 3. Validate uniqueness
            if self.validate_question(question_dict):
                print("Successfully generated unique question")
                # 4. Add to question_bank if unique
                self.question_bank.append(question_dict)
            else:
                print("Duplicate or invalid question detected.")

        return self.question_bank

    def validate_question(self, question: dict) -> bool:
        """
        Validates a quiz question for uniqueness within the generated quiz.
        """
        # 1. Ensure question has a "question" key
        if "question" not in question:
            return False

        # 2. Extract question text and check if it exists in self.question_bank
        question_text = question["question"].strip().lower()

        for q in self.question_bank:
            if "question" in q:
                existing_text = q["question"].strip().lower()
                if existing_text == question_text:
                    # Found a duplicate
                    return False

        # If no duplicates found, it's unique
        return True


# Test Generating the Quiz
if __name__ == "__main__":
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

        question_bank = None

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")

            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)

            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                st.write(topic_input)

                # Test the Quiz Generator
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question_bank = generator.generate_quiz()

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Questions:")
            for i, q in enumerate(question_bank, start=1):
                st.subheader(f"Question #{i}")
                st.json(q)
