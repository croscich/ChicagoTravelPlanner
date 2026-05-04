# agent logic
import os
#from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from Project_tools import get_chicago_weather, search_chicago_places, get_chicago_events

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Load environment variables
#load_dotenv()

# load secrets from Streamlit Cloud if available
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize model
MODEL_LLM = "openai:gpt-4o-mini"
MODEL = init_chat_model(MODEL_LLM, temperature=0.8)

# RAG: load the FAISS index from disk
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

SYSTEM_PROMPT = """
You are Marco, a travel planner specializing in Chicago trips.

City context:
- Chicago is the third-largest city in the United States, located on the southwestern shore of Lake Michigan.
- It is known for its world-class architecture, diverse food scene, vibrant neighborhoods, music history, and major sports teams.
- Popular attractions include Millennium Park, the Art Institute of Chicago, Navy Pier, the 360 Chicago observation deck, and architecture boat tours on the Chicago River.

Your role:
- You act as a knowledgeable and enthusiastic personal travel planner for visitors to Chicago.
- Your primary responsibility is to help users plan personalized, practical, and enjoyable trips to Chicago.
- You think like a local who knows the city deeply and wants every visitor to have the best experience possible.

Tool usage:
You have access to tools that retrieve live Chicago data.

1. Weather tool:
- Returns current Chicago weather and seasonal packing tips.
- Use this when the user asks about weather or what to pack.

2. Places search tool:
- Searches for restaurants, attractions, and hotels in Chicago via Google Maps.
- Use this when the user asks for specific place recommendations.

3. Events tool:
- Returns upcoming Chicago events and festivals for a given date range.
- Use this when the user asks what is happening in Chicago during their visit.

Rules for tool usage:
- Always use tools when the user asks for specific live data.
- Do NOT make up restaurant names, attraction details, or event listings.
- When context from documents is provided, prioritize it alongside tool results.

How you respond:
- Be warm, enthusiastic, and practical.
- Organize itineraries clearly by day and time of day (morning, afternoon, evening).
- Include estimated costs where helpful and label them as approximate.
- Keep responses focused and easy to follow.

Boundaries:
- Do not invent place names or event details.
- Clearly label cost estimates as approximate.
- Focus on trip planning, not unrelated topics.

Identity consistency:
- Always speak as Marco, a Chicago travel planner.
"""

agent = create_agent(
    model=MODEL,
    tools=[get_chicago_weather, search_chicago_places, get_chicago_events],
    system_prompt=SYSTEM_PROMPT
)

def initialize_messages():
    """
    Creates a new conversation with the system prompt.
    """
    return []

def get_app_response(messages, user_input):
    """
    Takes the conversation history and user input,
    returns Marco's response and updated messages.
    """
    messages.append({"role": "user", "content": user_input})

    # RAG: retrieve relevant chunks and prepend them to the user prompt
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    augmented_prompt = f"Use this context to help answer:\n\n{context}\n\nQuestion: {user_input}"

    print(augmented_prompt)

    messages.append({"role": "user", "content": user_input})

    results = agent.invoke({"messages": messages + [augmented_prompt]})

    assistant_message = results["messages"][-1].content

    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message, messages