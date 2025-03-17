from langchain_groq import ChatGroq
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")


MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""


CREATE_MEMORY_INSTRUCTION = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
- Personal details (name, location)
- Preferences (likes, dislikes)
- Interests and hobbies
- Past experiences
- Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""


def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace=namespace, key=key)

    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memory", user_id)
    key = 'user_memory'
    existing_memory = store.get(namespace, key=key)

    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'])

    store.put(namespace, key, {"memory": new_memory.content})


builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

across_thread_memory = InMemoryStore()
within_thread_memory = MemorySaver()

graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

config = {"configurable": {"thread_id": "1", "user_id": "1"}}

input_messages = ["Hi, my name is Harsh"]

res = graph.invoke({"messages": input_messages}, config, stream_mode="values")
print(res['messages'][1].content)

input_messages = ["I like to go to carting"]

for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


namespace = ("memory",'1')
existing_memory = across_thread_memory.get(namespace, "user_memory")
print(existing_memory.dict())
