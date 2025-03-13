from langchain_groq import ChatGroq
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

## -----ALl Prompts----- ##
subjects_prompt = """Generate a list of 5 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

## -----For Subjects Type----- ##
class Subjects(BaseModel):
    subjects: list[str]

## -----For Best Joke----- ##
class BestJoke(BaseModel):
    id: int

## -----OverallState that we will pass to Graph----- ##
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


## -----Node that creates subjects based on topic----- ##
def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = llm.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}

## -----Node that sends subjects to generate_joke node continuously----- ##
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

class JokeState(TypedDict):
    subject: str

class Joke(BaseModel):
    joke: str

## -----Node that generates jokes about subjects----- ##
def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = llm.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}

## -----This node will select best joke----- ##
def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = llm.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


builder = StateGraph(OverallState)

builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)

builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
# <--Above edge has condition that as long as there is subjects, continue_to_jokes will send that subject to generate_jokes node--> #
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)

graph = builder.compile()

for s in graph.stream({"topic": "Programming Languages"}):
    print(s)
