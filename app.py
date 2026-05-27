import streamlit as st
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Groq Agent", page_icon="🚀", layout="wide")
st.title("🦜 SALEHA's Agent")
st.caption("Chat with Groq + Tools")

# Get API key from Streamlit secrets (recommended for cloud)
if "groq_key" not in st.session_state:
    st.session_state.groq_key = st.secrets.get("GROQ_API_KEY", "")

if not st.session_state.groq_key:
    st.error("GROQ_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = st.session_state.groq_key

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"🌤️ The weather in {city} is sunny and 29°C right now!"

tools = [get_weather]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Simple memory using Streamlit session_state (works perfectly on cloud)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent = create_react_agent(llm, tools)
            result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
            response = result["messages"][-1].content
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.success("✅ Agent is live!")
st.sidebar.info("Model: llama-3.3-70b-versatile")
