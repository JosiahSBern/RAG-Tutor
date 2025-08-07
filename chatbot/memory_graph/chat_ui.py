import streamlit as st
import os
import json
from typing import List
from graph import (
    run_agent,
    recall_vector_store,
    get_user_id,
    load_all_user_memories,
    delete_user_memory_by_id,
    save_or_update_memory
)
from langchain_core.documents import Document

# Load all memories at startup ===
load_all_user_memories()

st.set_page_config(page_title="Quiztronics Tutor", layout="wide")

# User Login 
st.sidebar.title("User Login")
user_id = st.sidebar.text_input("Enter your student ID or alias:", "student_123").strip()

config = {
    "configurable": {
        "user_id": user_id
    }
}

# Sidebar Memory Viewer 
st.sidebar.markdown("---")
st.sidebar.subheader("Your Saved Memories")

def load_user_json_log(uid):
    path = f"user_memory_logs/{uid}.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def delete_from_json_log(uid, doc_id):
    path = f"user_memory_logs/{uid}.json"
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            if f'"{doc_id}"' not in line:
                f.write(line)

memory_docs = recall_vector_store.similarity_search(
    "", k=100, filter=lambda d: d.metadata.get("user_id") == user_id
)

for i, doc in enumerate(memory_docs):
    with st.sidebar.expander(f"Memory #{i+1}"):
        st.text_area("Content", doc.page_content, height=100, disabled=True)
        if st.button(f"❌ Delete", key=f"delete_{i}"):
            delete_user_memory_by_id(user_id, doc.id)
            delete_from_json_log(user_id, doc.id)
            st.rerun()


# Main Chat UI 
st.title("Quiztronics Tutor")
st.markdown("Ask your tutor a question!")

user_input = st.text_input("💬 Your message")

# Session tracking for exit chat
if "session_messages" not in st.session_state:
    st.session_state.session_messages = []

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        response = run_agent(user_input, user_id=user_id)

    st.session_state.session_messages.append(f"User: {user_input}")
    st.session_state.session_messages.append(f"Tutor: {response}")

    st.markdown("**Tutor says:**")
    st.success(response)

# Exit Chat & Summarize 
if st.button("Exit Chat"):
    if st.session_state.session_messages:
        full_chat = "\n".join(st.session_state.session_messages[-6:])  # Summarize last 6 lines
        summary_prompt = f"Summarize this chat to store as long-term memory:\n{full_chat}"

        with st.spinner("Summarizing and saving chat..."):
            try:
                result = save_recall_memory.invoke(summary_prompt, config)
                st.success("Session summary saved to memory.")
                st.toast(result)
                st.session_state.session_messages = []
            except Exception as e:
                st.error(f"Failed to save memory: {e}")
    else:
        st.warning("No messages to summarize.")
