"""LangGraph agent with branching logic: classify a topic as Technical or
Business, then generate the matching report.

    Classifier
      /      \
Technical   Business
     |          |
Tech Report  Biz Report
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class State(TypedDict):
    topic: str
    category: str
    report: str


def classifier(state: State) -> dict:
    """Classify the user's topic as 'technical' or 'business'."""
    prompt = (
        "Classify the following topic as exactly one word, either "
        "'technical' or 'business'. Respond with only that word.\n\n"
        f"Topic: {state['topic']}"
    )
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    if "technical" in category:
        state["category"] = "technical"
    else:
        state["category"] = "business"
    return {"category": state["category"]}


def route_by_category(state: State) -> Literal["technical", "business"]:
    return state["category"]


def technical_report(state: State) -> dict:
    """Generate a technical report for the topic."""
    prompt = (
        "Write a concise technical report on the following topic. "
        "Cover: overview, key technical details/architecture, challenges, "
        "and recommendations.\n\n"
        f"Topic: {state['topic']}"
    )
    response = llm.invoke(prompt)
    return {"report": response.content}


def business_report(state: State) -> dict:
    """Generate a business report for the topic."""
    prompt = (
        "Write a concise business report on the following topic. "
        "Cover: overview, market/financial implications, risks, "
        "and recommendations.\n\n"
        f"Topic: {state['topic']}"
    )
    response = llm.invoke(prompt)
    return {"report": response.content}


graph = (
    StateGraph(State)
    .add_node("classifier", classifier)
    .add_node("technical", technical_report)
    .add_node("business", business_report)

    .add_edge(START, "classifier")
    .add_conditional_edges(
        "classifier",
        route_by_category,
        {"technical": "technical", "business": "business"},
    )
    .add_edge("technical", END)
    .add_edge("business", END)
    .compile()
)


def save_mermaid_graph(path: str = "graph.mmd") -> None:
    """Write the graph's Mermaid diagram source to a file."""
    mermaid = graph.get_graph().draw_mermaid()
    with open(path, "w") as f:
        f.write(mermaid)


def save_mermaid_png(path: str = "graph.png") -> None:
    """Render the graph's Mermaid diagram to a PNG file."""
    with open(path, "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())


if __name__ == "__main__":
    save_mermaid_graph()
    save_mermaid_png()

    topic = input("Enter a topic: ").strip()
    result = graph.invoke({"topic": topic})
    print(f"\nCategory: {result['category']}")
    print(f"\nReport:\n{result['report']}")

### BUILD a LangGraph agent that that requires a parallel execution of tasks.  

### ADD LANGSMITH Instrumentation to the LangGraph and langchain agents to track the execution of the tasks.