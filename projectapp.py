import streamlit as st
from Project_step_5 import initialize_messages, get_app_response

company_logo = ("images/projectlogo.webp")
loading_icon = "images/loadingicon.webp"

st.set_page_config(page_title="Chicago Travel Planner", layout="wide")

st.image(company_logo)

st.title("Chicago Travel Planner")
st.subheader("Your Helper for Chicago Travel Plans")

# Initialize conversation memory once per session
# the conversation is initialized with the system prompt
# this code is using function initialize_messages() in the other file
if "messages" not in st.session_state:
    st.session_state.messages = initialize_messages()

# Display chat history (skip system message)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question...")
if user_input:
    # displays the new prompt as part of the conversation
    st.chat_message("user").write(user_input)

    with st.spinner("Planner is thinking..."):
        response, updated_messages = get_app_response(
            st.session_state.messages,
            user_input
    )

    # replace the session_state.messages with the updated list of messages
    st.session_state.messages = updated_messages
    # display the LLMs latest response
    st.chat_message("assistant", avatar=loading_icon).write(response)
