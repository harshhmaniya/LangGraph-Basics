from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, START, END

llm = ChatOllama(model="llama3.2")

def multiply(a: int, b: int) -> int:
    return a * b

llm_with_tools = llm.bind_tools([multiply])

class MessageState(MessagesState):
    pass

def tool_calling_llm(state: MessageState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessageState)
builder.add_node("tool_calling_llm", tool_calling_llm)

builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

result = graph.invoke({"messages": "What is 2 multiplied by 3"})
print(result)
