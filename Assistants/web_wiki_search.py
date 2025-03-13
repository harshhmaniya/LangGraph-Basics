from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from typing import Annotated
import operator
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]


def search_web(state):
    """ Retrieve docs from web search """

    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipedia(state):
    """ Retrieve docs from wikipedia """

    search_docs = WikipediaLoader(query=state['question'],
                                  load_max_docs=2).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def generate_answer(state):
    """ Node to answer a question """

    context = state["context"]
    question = state["question"]

    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question,
                                                 context=context)

    answer = llm.invoke([SystemMessage(content=answer_instructions)] + [HumanMessage(content="Answer the question.")])

    return {"answer": answer}


builder = StateGraph(State)

builder.add_node("Web_Search", search_web)
builder.add_node("Wikipedia_Search", search_wikipedia)
builder.add_node("Generate_Answer", generate_answer)

builder.add_edge(START, "Web_Search")
builder.add_edge(START, "Wikipedia_Search")
builder.add_edge("Web_Search", "Generate_Answer")
builder.add_edge("Wikipedia_Search", "Generate_Answer")
builder.add_edge("Generate_Answer", END)

graph = builder.compile()

response = graph.invoke({"question": "What is Machine Learning?"})
print(response['answer'].content)
