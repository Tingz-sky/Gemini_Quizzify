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

        # Suppose your ChromaCollectionCreator has self.db = Chroma(...) inside
        retriever = self.vectorstore.db.as_retriever()

        prompt = PromptTemplate.from_template(self.system_template)

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )

        chain = setup_and_retrieval | prompt | self.llm
        response = chain.invoke(self.topic)
        return response

    def generate_quiz(self) -> list:
        """
        Generates a list of unique quiz questions based on the specified topic and number of questions.
        """
        self.question_bank = []  # Reset the question bank

        for _ in range(self.num_questions):
            # 1) Generate a question (JSON string).
            question_str = self.generate_question_with_vectorstore()

            # 2) Attempt to parse JSON
            try:
                question_dict = json.loads(question_str)
            except json.JSONDecodeError:
                print("Failed to decode question JSON.")
                continue

            # 3) Validate uniqueness
            if self.validate_question(question_dict):
                print("Successfully generated unique question")
                self.question_bank.append(question_dict)
            else:
                print("Duplicate or invalid question detected.")

        return self.question_bank

    def validate_question(self, question: dict) -> bool:
        """
        Validates a quiz question for uniqueness within the generated quiz.
        """
        if "question" not in question:
            return False

        question_text = question["question"].strip().lower()
        for q in self.question_bank:
            if "question" in q:
                existing_text = q["question"].strip().lower()
                if existing_text == question_text:
                    return False
        return True


###################### QUIZ MANAGER ##########################
class QuizManager:
    ##########################################################
    def __init__(self, questions: list):
        """
        Task: Initialize the QuizManager class with a list of quiz questions.
        """
        # 1) Store the provided list in an instance variable
        self.questions = questions
        # 2) Calculate the total number
        self.total_questions = len(questions)

    ##########################################################

    def get_question_at_index(self, index: int):
        """
        Retrieves the quiz question object at the specified index,
        wrapping around if out of bounds.
        """
        valid_index = index % self.total_questions
        return self.questions[valid_index]

    ##########################################################
    def next_question_index(self, direction=1):
        """
        Adjust the current quiz question index based on the specified direction.
        """
        if "question_index" not in st.session_state:
            st.session_state["question_index"] = 0

        current_index = st.session_state["question_index"]
        # Move forward or backward
        new_index = (current_index + direction) % self.total_questions
        st.session_state["question_index"] = new_index
    ##########################################################


# Test Generating the Quiz + Using the QuizManager
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

        embed_client = EmbeddingClient(**embed_config)
        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        question_bank = None

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for ingestion, the topic for the quiz, and click Generate!")

            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions_num = st.slider("Number of Questions", min_value=1, max_value=10, value=1)

            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                st.write(f"Generating {questions_num} questions for topic: {topic_input}")

                generator = QuizGenerator(topic_input, questions_num, chroma_creator)
                question_bank = generator.generate_quiz()

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question:")

            # -- Initialize QuizManager
            quiz_manager = QuizManager(question_bank)

            # Make sure "question_index" is in session_state
            if "question_index" not in st.session_state:
                st.session_state["question_index"] = 0

            # Task 9: Format & display the question
            with st.form("Multiple Choice Question"):
                index_question = quiz_manager.get_question_at_index(st.session_state["question_index"])

                choices = []
                for c in index_question['choices']:
                    # c is like {"key": "A", "value": "some answer"}
                    key = c["key"]
                    value = c["value"]
                    choices.append(f"{key}) {value}")

                # Display question on Streamlit
                st.subheader(index_question["question"])
                answer = st.radio(
                    'Choose the correct answer',
                    choices
                )

                submitted_mc = st.form_submit_button("Submit")
                if submitted_mc:
                    correct_answer_key = index_question["answer"]
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")

            # Add buttons to navigate next/previous question
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous Question"):
                    quiz_manager.next_question_index(direction=-1)
                    st.experimental_rerun()
            with col2:
                if st.button("Next Question"):
                    quiz_manager.next_question_index(direction=1)
                    st.experimental_rerun()
