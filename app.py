import streamlit as st
import os
import sqlite3
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

st.set_page_config(page_title="Groq Agent", page_icon="🚀", layout="wide")
st.title("🦜 Groq LangChain Agent with Permanent Memory")
st.caption("Remembers everything — even after you close the app!")

# API Key
if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""

groq_key = st.text_input("Enter your Groq API Key", type="password", value=st.session_state.groq_key)
if groq_key:
    st.session_state.groq_key = groq_key
    os.environ["GROQ_API_KEY"] = groq_key

if not os.environ.get("GROQ_API_KEY"):
    st.warning("Please enter your Groq API key above 👆")
    st.stop()

# Tool
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"🌤️ The weather in {city} is sunny and 29°C right now!"

tools = [get_weather]

# LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# === Permanent SQLite Memory ===
if "checkpointer" not in st.session_state:
    # Create persistent SQLite connection
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    st.session_state.checkpointer = SqliteSaver(conn)

# Create agent with permanent memory
if "agent" not in st.session_state:
    st.session_state.agent = create_react_agent(
        llm,
        tools,
        checkpointer=st.session_state.checkpointer
    )

# Persistent thread (one conversation forever)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "permanent_chat_1"

# Chat display
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )
            response = result["messages"][-1].content
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.success("✅ Permanent Memory Enabled (SQLite)")
st.sidebar.info("Model: llama-3.3-70b-versatile")
st.sidebar.caption("All conversations saved in checkpoints.db")