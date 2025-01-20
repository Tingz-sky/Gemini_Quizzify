import streamlit as st
import os
import sys
import json

sys.path.append(os.path.abspath('../../'))
from File_uploader import DocumentProcessor
from vertex_embedding import EmbeddingClient
from integration import ChromaCollectionCreator
from quiz_algo import QuizGenerator  # from tasks.task_8.task_8 -> now quiz_generator
from ui import QuizManager  # from tasks.task_9.task_9 -> now quiz_manager

if __name__ == "__main__":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/liuhaibo/gcloud_key/geminisample-425519-a9f2d1d62e3b.json"
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "location": "us-central1"
    }

    # ----------------- Add Session State -----------------
    # If we haven't stored a question_bank or it's empty, we do the "Quiz Builder" form.
    if 'question_bank' not in st.session_state or len(st.session_state['question_bank']) == 0:

        # Step 1: init the question bank list in st.session_state
        if 'question_bank' not in st.session_state:
            st.session_state['question_bank'] = []
        if 'display_quiz' not in st.session_state:
            st.session_state['display_quiz'] = False
        if 'question_index' not in st.session_state:
            st.session_state['question_index'] = 0

        screen = st.empty()
        with screen.container():
            st.header("Quiz Builder")

            # Create a new st.form flow control for Data Ingestion
            with st.form("Load Data to Chroma"):
                st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")

                processor = DocumentProcessor()
                processor.ingest_documents()

                embed_client = EmbeddingClient(**embed_config)

                chroma_creator = ChromaCollectionCreator(processor, embed_client)

                # Step 2: Set topic input and number of questions
                topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
                questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)

                submitted = st.form_submit_button("Submit")

                if submitted:
                    chroma_creator.create_chroma_collection()

                    if len(processor.pages) > 0:
                        st.write(f"Generating {questions} questions for topic: {topic_input}")

                    # Step 3: Initialize a QuizGenerator class using the topic, number of questions, and the chroma collection
                    generator = QuizGenerator(topic_input, questions, chroma_creator)

                    question_bank = generator.generate_quiz()

                    # Step 4: Initialize the question bank list in st.session_state
                    st.session_state['question_bank'] = question_bank
                    # Step 5: Set a display_quiz flag in st.session_state to True
                    st.session_state['display_quiz'] = True
                    # Step 6: Set the question_index to 0 in st.session_state
                    st.session_state['question_index'] = 0

    # ----------------- Quiz Display Screen -----------------
    elif st.session_state["display_quiz"]:

        st.empty()
        with st.container():
            st.header("Generated Quiz Question: ")

            quiz_manager = QuizManager(st.session_state['question_bank'])

            # Format the question and display it
            with st.form("MCQ"):
                # Step 7: Set index_question using the Quiz Manager method get_question_at_index passing the st.session_state["question_index"]
                index_question = quiz_manager.get_question_at_index(st.session_state["question_index"])

                # Unpack choices for radio button
                choices = []
                for choice in index_question['choices']:
                    key = choice['key']
                    value = choice['value']
                    choices.append(f"{key}) {value}")

                # Display the Question
                st.write(f"{st.session_state['question_index'] + 1}. {index_question['question']}")
                answer = st.radio(
                    "Choose an answer",
                    choices,
                    index=None
                )

                answer_choice = st.form_submit_button("Submit")

                # Step 8: Next/Previous Buttons
                # We can place them outside or inside this form, but typically separate is easier
                # We'll show you how to do it inside the form for clarity:

                col1, col2 = st.columns(2)
                with col1:
                    go_next = st.form_submit_button("Next Question",
                                                    on_click=lambda: quiz_manager.next_question_index(direction=1))
                with col2:
                    go_prev = st.form_submit_button("Previous Question",
                                                    on_click=lambda: quiz_manager.next_question_index(direction=-1))

                if answer_choice and answer is not None:
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
                    st.write(f"Explanation: {index_question['explanation']}")
