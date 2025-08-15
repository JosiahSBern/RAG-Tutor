import json
import os
import uuid
from typing import List, Literal, Optional
# AIzaSyB--kGwDbI5ZUNHfHqlt2ckARw8K4lEm8g
# Core imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.utils import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from datetime import datetime


# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately

# Model imports - using newer langchain-ollama
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer

# Embeddings - using local HuggingFace embeddings to avoid OpenAI dependency
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize model and tokenizer
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set your Gemini API key (use env var or config)
os.environ["GOOGLE_API_KEY"] = "AIzaSyB--kGwDbI5ZUNHfHqlt2ckARw8K4lEm8g"
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Local vector storage for memory
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
recall_vector_store = InMemoryVectorStore(embeddings)

def get_user_id(config: RunnableConfig) -> str:
    """Gets unique user id"""
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")
    return user_id

# Memory recall and searching tools
@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore and JSON log file for persistence and audit."""
    user_id = config["configurable"].get("user_id")
    if not user_id:
        raise ValueError("User ID required to store memory.")

    # Create document for vectorstore
    document = Document(
        page_content=memory,
        id=str(uuid.uuid4()),
        metadata={"user_id": user_id}
    )

    # Save to vectorstore for semantic search
    recall_vector_store.add_documents([document])

    # Also log to JSON for audit
    os.makedirs("user_memory_logs", exist_ok=True)
    log_path = f"user_memory_logs/{user_id}.json"

    memory_entry = {
        "timestamp": str(datetime.now()),
        "id": document.id,
        "memory": memory,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        json.dump(memory_entry, f)
        f.write("\n")

    return f"Memory saved to vectorstore and logged for user {user_id}."

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories"""
    user_id = get_user_id(config)
    
    def __filter_function(doc: Document) -> bool:
        """Filters relevant information in memory"""
        return doc.metadata.get("user_id") == user_id
    
    documents = recall_vector_store.similarity_search(
        query, k=3, filter=__filter_function
    )
    return [document.page_content for document in documents]




# Define tools list
tools = [save_recall_memory, search_recall_memories]

def delete_user_memory_by_id(user_id: str, doc_id: str):
    """Delete from vectorstore and JSON."""
    if not doc_id:
        print("⚠️ No document ID provided for deletion.")
        return

    try:
        recall_vector_store.delete([doc_id])  # This expects a list of IDs
        print(f"✅ Deleted from vectorstore: {doc_id}")
    except Exception as e:
        print(f"❌ Error deleting from vectorstore: {e}")

    # Now delete from JSON log
    path = f"user_memory_logs/{user_id}.json"
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            if f'"{doc_id}"' not in line:
                f.write(line)



class State(MessagesState):
    """Add memories that will be retrieved based on the conversation context"""
    recall_memories: List[str]

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
📚 You are a supportive and memory-aware AI tutor designed to help C++ students, especially those in CSC211.

🎯 Your job:
- Teach C++ concepts clearly and kindly.
- Remember important details about the student.
- Use tools to store and retrieve helpful information across sessions.

🧠 Your tools:
1. save_recall_memory(memory: str)
   - Use this to save any valuable student information, including:
     - struggles or confusion ("I'm stuck on loops")
     - learning goals ("I want to master functions")
     - preferences ("I prefer learning by examples")
     - progress ("I finally understood arrays")
     - identity ("My name is Alex", "I'm a CS major")

2. search_recall_memories(query: str)
   - Use this to recall relevant info when a student asks something you’ve seen before.

📌 How to use a tool:
TOOL_CALL: tool_name("your input here")

🔁 Memory Policy:
- Save *early and often*
- If you’re not sure — SAVE IT ANYWAY.
- Summarize what the student said in a short sentence when saving.

✅ Example 1:
User: "I'm having a tough time with pointers and memory addresses."
You: TOOL_CALL: save_recall_memory("Student is struggling with pointers and memory addresses in C++")

✅ Example 2:
User: "My name is Janelle and I'm aiming for an A in CSC211."
You: TOOL_CALL: save_recall_memory("Student's name is Janelle and wants an A in CSC211")

✅ Example 3:
User: "Hey, I finally figured out how to write a recursive function!"
You: TOOL_CALL: save_recall_memory("Student has successfully learned how to write recursive functions")

📂 What you remember (recall_memories):
{recall_memories}

💬 Speak in a friendly, helpful tone. Encourage the student.
Always remember to SAVE helpful information!
"""
    ),
    ("placeholder", "{messages}")
])



def agent(state: State, config: RunnableConfig) -> State:
    """Process the current state and generate a response using the LLM."""
    recall_str = (
        "<recall_memory>\n" + "\n".join(state.get("recall_memories", [])) + "\n</recall_memory>"
    )
    
    # Create enhanced prompt with tool instructions
    enhanced_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. You have access to two tools:\n"
            "1. save_recall_memory(memory: str) - Save important information\n"
            "2. search_recall_memories(query: str) - Search for relevant memories\n\n"
            "When you need to use a tool, format it as:\n"
            "TOOL_CALL: tool_name(arguments)\n\n"
            "Memory Usage Guidelines:\n"
            "- Save important user information, preferences, and context\n"
            "- Search for relevant memories to provide personalized responses\n"
            "- Build a comprehensive understanding of the user over time\n\n"
            "## Current Recall Memories\n"
            "{recall_memories}\n\n"
            "Engage naturally and use tools when helpful for remembering or recalling information.",
        ),
        ("placeholder", "{messages}"),
    ])
    
    # Get response from model
    bound = enhanced_prompt | model
    prediction = bound.invoke({
        "messages": state["messages"],
        "recall_memories": recall_str,
    }, config)
    
    # Check if the response contains tool calls
    response_content = prediction.content
    tool_calls = []
    
    # Parse tool calls from response
    if "TOOL_CALL:" in response_content:
        lines = response_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith("TOOL_CALL:"):
                tool_call_str = line.replace("TOOL_CALL:", "").strip()
                
                # Parse save_recall_memory calls
                if tool_call_str.startswith("save_recall_memory("):
                    memory_content = tool_call_str[len("save_recall_memory("):-1]
                    memory_content = memory_content.strip('"\'')
                    try:
                        result = save_recall_memory.invoke(memory_content, config)
                        tool_calls.append(f"Tool executed: {result}")
                    except Exception as e:
                        tool_calls.append(f"Tool error: {e}")
                
                # Parse search_recall_memories calls
                elif tool_call_str.startswith("search_recall_memories("):
                    query_content = tool_call_str[len("search_recall_memories("):-1]
                    query_content = query_content.strip('"\'')
                    try:
                        result = search_recall_memories.invoke(query_content, config)
                        tool_calls.append(f"Found memories: {result}")
                    except Exception as e:
                        tool_calls.append(f"Tool error: {e}")
    
    # If tools were called, add results to the response
    if tool_calls:
        # Remove TOOL_CALL lines from the visible message
        cleaned = "\n".join(
            line for line in response_content.split("\n")
            if not line.strip().startswith("TOOL_CALL:")
        ).strip()
        prediction.content = cleaned + ("\n\n" + "\n".join(tool_calls) if tool_calls else "")

    
    return {"messages": [prediction]}

def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation."""
    convo_str = get_buffer_string(state["messages"])
    # Truncate to 2048 tokens
    convo_str = convo_str[:4096]
    
    try:
        recall_memories = search_recall_memories.invoke(convo_str, config)
    except Exception as e:
        print(f"Error loading memories: {e}")
        recall_memories = []
    
    return {"recall_memories": recall_memories}

def route_tools(state: State):
    """Always go to END since we handle tools inline now."""
    return END

# Short-term memory summarizer
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

def condense_to_long_term(state: State, config: RunnableConfig) -> State:
    """Save short-term summary to long-term memory if it's meaningful."""
    summary = state.get("short_term_summary", "").strip()

    # Optional: Add filtering or confidence checking here
    if summary and len(summary.split()) > 3:  # Don't store tiny/empty thoughts
        try:
            result = save_recall_memory.invoke(summary, config)
            print(f"[Memory Saved]: {result}")
        except Exception as e:
            print(f"[Memory Error]: {e}")

    return {}  # No changes to state




builder = StateGraph(State)

builder.add_node("load_memories", load_memories)
builder.add_node("summarize", summarization_node)
builder.add_node("agent", agent)
builder.add_node("condense", condense_to_long_term)

builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "summarize")
builder.add_edge("summarize", "agent")
builder.add_edge("agent", "condense")
builder.add_edge("condense", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)



def load_json_memories_to_vectorstore(user_id: str):
    path = f"user_memory_logs/{user_id}.json"
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                memory_data = json.loads(line)
                document = Document(
                    page_content=memory_data["memory"],
                    id=memory_data["id"],
                    metadata={"user_id": user_id}
                )
                recall_vector_store.add_documents([document])
            except json.JSONDecodeError:
                continue  # Skip bad lines

# Function to run the agent
def run_agent(user_input: str, user_id: str = "default_user", thread_id: str = "default_thread"):
    """Run the agent with a user input"""
    config = {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id
        }
    }
    
    # Create the input state
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "recall_memories": []
    }
    
    # Run the graph
    final_state = graph.invoke(initial_state, config)
    
    # Return the last AI message
    return final_state["messages"][-1].content

def load_all_user_memories():
    folder = "user_memory_logs"
    if not os.path.exists(folder):
        return

    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            user_id = filename.replace(".json", "")
            load_json_memories_to_vectorstore(user_id)


from typing import List

def exit_chat_summary(user_id: str, messages: List[str]):
    """
    Summarizes the last few messages of a session and saves them to memory.
    """
    if not messages:
        return "No messages to summarize."

    # Take the last 6 messages for summarization context
    summary_input = "\n".join(messages[-6:])  
    summary_prompt = f"Summarize this chat into one sentence to store as memory:\n{summary_input}"

    try:
        # Generate summary using the model
        summary = model.invoke(summary_prompt).content.strip()

        # Save the summary to memory
        config = {
            "configurable": {
                "user_id": user_id
            }
        }
        result = save_recall_memory.invoke(summary, config)
        return result
    except Exception as e:
        return f"Error saving chat summary: {e}"


# Example usage
if __name__ == "__main__":
    # Test the agent
    response1 = run_agent("Hi, my name is Alice and I love Python programming!")
    print("Response 1:", response1)
    
    response2 = run_agent("What do you remember about me?")
    print("Response 2:", response2)