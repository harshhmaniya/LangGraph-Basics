import random
from typing import TypedDict,Literal
from IPython.display import display, Image
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    graph_state: str

def node_1(state):
    print("--Node 1--")
    return {"graph_state": state['graph_state'] + "I am"}

def node_2(state):
    print("--Node 2--")
    return {"graph_state": state['graph_state'] + " Happy!"}

def node_3(state):
    print("--Node 3--")
    return {"graph_state": state['graph_state'] + " Sad!"}

def decide_mood(state) -> Literal["node_2", "node_3"]:
    print("--Decide Mood--")
    return "node_2" if random.random() < 0.5 else "node_3"

builder = StateGraph(State)
builder.add_node("node_1",node_1)
builder.add_node("node_2",node_2)
builder.add_node("node_3",node_3)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

response = graph.invoke({"graph_state": "My Name is Harsh Maniya."})
print(response)
