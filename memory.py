from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

def multiply(a: int, b: int) -> int:
    """
    Multiply two integers and return the result.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    return a * b

def addition(a: int, b: int) -> int:
    """
    Add two integers and return the result.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    return a + b


def divide(a: int, b: int) -> float:
    """
    Divide two integers and return the result.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The division of a and b.
    """
    return a / b

llm_with_tools = llm.bind_tools([multiply, addition, divide])

sys_msg = SystemMessage(content="You are a helpful assistant that performs arithmetic operations.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([multiply, addition, divide]))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id" : "1"
    }
}

messages = [HumanMessage(content="First divide 100 by 5 then add 10 in it and then multiply it by 2")]
messages = react_graph_memory.invoke({"messages": messages}, config=config)

for m in messages['messages']:
    m.pretty_print()

messages = [HumanMessage(content="Add 20 to that")]
messages = react_graph_memory.invoke({"messages": messages}, config=config)

for m in messages['messages']:
    m.pretty_print()