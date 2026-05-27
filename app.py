import streamlit as st
import os
import sqlite3
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

st.set_page_config(page_title="Groq Agent", page_icon="🚀", layout="wide")
st.title("🦜 Groq LangChain Agent")
st.caption("💾 Permanent Memory + Faster Model")

# API Key check
if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"🌤️ The weather in {city} is sunny and 29°C right now!"

tools = [get_weather]

# Changed to faster model with higher free-tier limits
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Permanent Memory
if "checkpointer" not in st.session_state:
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    st.session_state.checkpointer = SqliteSaver(conn)

if "agent" not in st.session_state:
    st.session_state.agent = create_react_agent(
        llm, 
        tools, 
        checkpointer=st.session_state.checkpointer
    )

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "permanent_chat_1"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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

st.sidebar.success("✅ Permanent Memory Active")
st.sidebar.info("Model: llama-3.1-8b-instant (Fast + High Limits)")
