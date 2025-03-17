import uuid
from langchain_groq import ChatGroq
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
load_dotenv()

in_memory_store = InMemoryStore()

user_id = "1"
namespace_for_memory = (user_id, "memories")

# Save a memory to namespace as key and value
key = str(uuid.uuid4())

# The value needs to be a dictionary
value = {"food_preference" : "I like pizza"}

# Save the memory
in_memory_store.put(namespace_for_memory, key, value)

memories = in_memory_store.search(namespace_for_memory)
print(type(memories))

print(memories[0].dict())

# Get the memory by namespace and key
memory = in_memory_store.get(namespace_for_memory, key)
print(memory.dict())