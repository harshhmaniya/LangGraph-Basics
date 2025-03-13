from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END, MessagesState
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

def filter_message(state: MessagesState): # keeps last 2 messages and deletes everything else
    remaining_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": remaining_messages}

def chat_model_node(state: MessagesState):
    return {"messages": llm.invoke(state['messages'])}

# Alternate Approach (in the llm calling itself we can just take last 2 messages)
# If we go for this technique then we have to eliminate filter message node and edge
def chat_model(state: MessagesState):
    return {"messages": llm.invoke(state['messages'][-2:])}


builder = StateGraph(MessagesState)

builder.add_node("chat_model", chat_model_node)
builder.add_node("filter_message", filter_message)

builder.add_edge(START, "filter_message")
builder.add_edge("filter_message", "chat_model")
builder.add_edge("chat_model", END)

graph = builder.compile()

messages = [AIMessage("Hi.", name="Bot", id="1"),
            HumanMessage("Hi.", name="Harsh", id="2"),
            AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"),
            HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Harsh", id="4")]

output = graph.invoke({'messages': messages})
for message in output['messages']:
    message.pretty_print()