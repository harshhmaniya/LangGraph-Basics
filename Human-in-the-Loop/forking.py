from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine the control flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

graph = builder.compile(checkpointer=MemorySaver())

initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

thread = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

# State history
all_states = [s for s in graph.get_state_history(thread)]
print("Total states -->", len(all_states))

# State where we got human input
to_replay = all_states[-2]
# print(to_replay)

print(to_replay.values)

# Next node to call
print("Next Node to call -->", to_replay.next)

# Config of the state
print("Config of the state -->", to_replay.config)

# Replay from the current checkpoint (that is to_replay.config)
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    event['messages'][-1].pretty_print()



# --------Forking-------- #

# When we want to run from that same step, but with a different input. --> That is called as Forking

to_fork = all_states[-2]
print("Message We want to fork -->", to_fork.values["messages"])

# Config of the state that we want to fork
print("Config -->",to_fork.config)

# Here when updating state we will pass message ID of old message, that way we can overwrite old message
# If we do not pass message ID then old message will not be overwritten, but new message will appended at the end of list
fork_config = graph.update_state(
    to_fork.config,
    {"messages": [HumanMessage(content='Multiply 5 and 3',
                               id=to_fork.values["messages"][0].id)]},
)
print("Config of forked state -->", fork_config)

all_states = [state for state in graph.get_state_history(thread) ]
print(all_states[0].values["messages"])

for event in graph.stream(None, fork_config, stream_mode="values"):
    event['messages'][-1].pretty_print()
