import os
import streamlit as st
import vertexai
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part, Content, ChatSession

# Set the environment variable for the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/liuhaibo/gcloud_key/geminisample-425519-a9f2d1d62e3b.json"

# Use the correct project ID
project = "geminisample-425519"
vertexai.init(project=project, location="us-central1")

# Configure the generation settings for the model
config = generative_models.GenerationConfig(
    temperature=0.4
)

# Create a GenerativeModel instance with the specified configuration
model = generative_models.GenerativeModel(
    "gemini-pro",
    generation_config=config
)

# Start a chat session with the model, skipping response validation
chat = model.start_chat(response_validation=False)

def llm_function(chat: ChatSession, query):
    response = chat.send_message(query)
    output = response.candidates[0].content.parts[0].text
    with st.chat_message("model"):
        st.markdown(output)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    st.session_state.messages.append(
        {
            "role": "model",
            "content": output
        }
    )

# Initialize session state if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Gemini Explorer")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new query
query = st.chat_input("Enter your message to Gemini:")

if query:
    llm_function(chat, query)

# Capture user information
user_name = st.text_input("Please enter your name")

# Implement Personalized Greetings
if len(st.session_state.messages) == 0:
    if user_name:
        personalized_prompt = f"Ahoy, {user_name}! I'm ReX, your interactive assistant powered by Google Gemini. Let's chat with emojis!"
    else:
        personalized_prompt = "Ahoy there! I'm ReX, your interactive assistant powered by Google Gemini. Let's chat with emojis!"
    llm_function(chat, personalized_prompt)

# # Initial introduction message
# if len(st.session_state.messages) == 0:
#     initial_prompt = "Introduce yourself as ReX, an assistant powered by Google Gemini. You use emojis to be interactive"
#     llm_function(chat, initial_prompt)
