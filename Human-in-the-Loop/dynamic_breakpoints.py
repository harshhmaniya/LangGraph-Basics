from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
load_dotenv()


class State(TypedDict):
    input: str


def step_1(state: State) -> State:
    print("---Step 1---")
    return state


def step_2(state: State) -> State:
    # Let's optionally raise a NodeInterrupt if the length of the input is longer than 5 characters
    if len(state['input']) > 5:
        raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")

    print("---Step 2---")
    return state


def step_3(state: State) -> State:
    print("---Step 3---")
    return state

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

initial_input = {"input": "hello world"}
thread_config = {"configurable": {"thread_id": "1"}}

# Whole graph will not run because input is greater than 5 charcters
for event in graph.stream(initial_input, thread_config, stream_mode="values"):
    print(event)

state = graph.get_state(thread_config)
print("Next State -> ", state.next)
print(state.tasks)

# Update state
graph.update_state(
    thread_config,
    {"input": "hi"},
)

for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)