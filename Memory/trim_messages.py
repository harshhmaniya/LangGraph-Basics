from langchain_core.messages import trim_messages
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph, MessagesState
from dotenv import load_dotenv
load_dotenv()

# Trimming is used in very long conversations

llm = ChatGroq(model="llama-3.3-70b-versatile")

def chat_model_node(state: MessagesState):
    messages = trim_messages(
            messages=state["messages"],                              # Messages to be passed
            include_system=True,                                     # First System Message will be included or not
            max_tokens=100,                                          # Max tokens ww will allow
            strategy="last",                                         # We want to keep recent last messages or first messages
            token_counter=ChatGroq(model="llama-3.3-70b-versatile"), # Token counter -> can be a function or llm
            allow_partial=False,                                     # Partial messages will be allowed or not when trimming hits max_tokens
        )
    return {"messages": [llm.invoke(messages)]}

builder = StateGraph(MessagesState)

builder.add_node("chat_model", chat_model_node)

builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)

graph = builder.compile()
