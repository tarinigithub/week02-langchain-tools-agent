"""LangGraph agent with parallel execution: an orchestrator plans three
sections based on the topic's type, three workers write them at the same time,
and an aggregator merges the results into one report.

        Orchestrator
       /     |      \
  Worker 1  Worker 2  Worker 3      (run in parallel)
       \     |      /
         Aggregator
"""

import operator
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# Section briefs per topic type: the orchestrator picks one set, and the three
# workers below write one section each, in parallel.
SECTION_PLANS = {
    "technical": [
        "Architecture and how it works",
        "Implementation challenges and trade-offs",
        "Recommendations and next steps",
    ],
    "business": [
        "Market overview and opportunity",
        "Financial implications and risks",
        "Recommendations and next steps",
    ],
}


class TopicInput(TypedDict):
    """Graph input: the topic is the only thing the caller supplies.

    Declaring it as `input_schema` keeps the graph's working keys out of the
    input, so LangGraph Studio asks for a topic instead of all of them.
    """

    topic: str


class State(TypedDict):
    topic: str
    category: str
    plan: list[str]
    # Workers run in parallel and all write to `sections`, so the reducer
    # (operator.add) concatenates their results instead of overwriting.
    sections: Annotated[list[tuple[int, str]], operator.add]
    report: str


def orchestrator(state: State) -> dict:
    """Classify the topic type and choose the three sections to write."""
    prompt = (
        "Classify the following topic as exactly one word, either "
        "'technical' or 'business'. Respond with only that word.\n\n"
        f"Topic: {state['topic']}"
    )
    answer = llm.invoke(prompt).content.strip().lower()
    category = "technical" if "technical" in answer else "business"
    return {"category": category, "plan": SECTION_PLANS[category]}


def write_section(state: State, index: int) -> dict:
    """Write one section of the report; shared by all three workers."""
    heading = state["plan"][index]
    prompt = (
        f"Write the '{heading}' section of a {state['category']} report on the "
        "topic below. Write 3-5 sentences of prose, no heading, no preamble.\n\n"
        f"Topic: {state['topic']}"
    )
    return {"sections": [(index, llm.invoke(prompt).content.strip())]}


def worker_1(state: State) -> dict:
    return write_section(state, 0)


def worker_2(state: State) -> dict:
    return write_section(state, 1)


def worker_3(state: State) -> dict:
    return write_section(state, 2)


def aggregator(state: State) -> dict:
    """Merge the workers' sections, in plan order, into the final report."""
    body = "\n\n".join(
        f"## {state['plan'][index]}\n\n{text}"
        for index, text in sorted(state["sections"])
    )
    return {"report": f"# {state['category'].title()} report: {state['topic']}\n\n{body}"}


graph = (
    StateGraph(State, input_schema=TopicInput)
    .add_node("orchestrator", orchestrator)
    .add_node("worker_1", worker_1)
    .add_node("worker_2", worker_2)
    .add_node("worker_3", worker_3)
    .add_node("aggregator", aggregator)

    .add_edge(START, "orchestrator")
    # Three edges out of one node = the workers run in parallel...
    .add_edge("orchestrator", "worker_1")
    .add_edge("orchestrator", "worker_2")
    .add_edge("orchestrator", "worker_3")
    # ...and the aggregator waits for all three before it runs.
    .add_edge("worker_1", "aggregator")
    .add_edge("worker_2", "aggregator")
    .add_edge("worker_3", "aggregator")
    .add_edge("aggregator", END)
    .compile()
)


def save_mermaid_graph(path: str = "parallel_graph.mmd") -> None:
    """Write the graph's Mermaid diagram source to a file."""
    with open(path, "w") as f:
        f.write(graph.get_graph().draw_mermaid())


def save_mermaid_png(path: str = "parallel_graph.png") -> None:
    """Render the graph's Mermaid diagram to a PNG file."""
    with open(path, "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())


if __name__ == "__main__":
    save_mermaid_graph()
    save_mermaid_png()

    topic = input("Enter a topic: ").strip()
    result = graph.invoke({"topic": topic})
    print(f"\nCategory: {result['category']}")
    print(f"\n{result['report']}")
