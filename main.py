import os
from functools import cache

import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

from ingest import EMBED_MODEL, INDEX_NAME

load_dotenv()


@cache
def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=PineconeEmbeddings(model=EMBED_MODEL),
    )


def search_apple_10k(query: str) -> str:
    """Search Apple's 2025 Form 10-K annual report for financial and business
    information, e.g. revenue, net sales, net income, risks or segments."""
    docs = get_vectorstore().similarity_search(query, k=4)
    if not docs:
        return "No matching pages found in the Apple 10-K."
    return "\n\n---\n\n".join(
        f"[Page {int(d.metadata['page'])}]\n{d.page_content}" for d in docs
    )


def get_stock_price(ticker: str) -> str:
    """Get the last closing stock price for a given ticker symbol, e.g. AAPL."""
    stock = yf.Ticker(ticker)
    price = stock.fast_info.get("lastPrice")
    if price is None:
        return f"Could not find a price for ticker '{ticker}'."
    return f"The last price for {ticker.upper()} is ${price:.2f}"


def get_weather(city: str) -> str:
    """Get the current weather for a given city name, e.g. London."""
    api_key = os.environ.get("WEATHER_API_KEY")
    if not api_key:
        return "WEATHER_API_KEY is not set."

    response = requests.get(
        "https://api.weatherapi.com/v1/current.json",
        params={"key": api_key, "q": city},
        timeout=10,
    )
    if response.status_code != 200:
        return f"Could not find weather for '{city}'."

    data = response.json()
    description = data["current"]["condition"]["text"].lower()
    temp = data["current"]["temp_c"]
    return f"The weather in {city} is {description} at {temp}°C."


model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

agent = create_agent(
    model=model,
    tools=[get_stock_price, get_weather, search_apple_10k],
    system_prompt=(
        "You are a helpful assistant with access to a stock price tool "
        "(`get_stock_price`), a weather tool (`get_weather`) and a search tool "
        "over Apple's 2025 Form 10-K (`search_apple_10k`). Decide which "
        "tool to use based on the user's query, using several in sequence if "
        "the query needs them. For questions about Apple's financials, answer "
        "only from the 10-K search results and cite the page number. Always "
        "give a clear, concise answer after using the tools."
    ),
)


def main() -> None:
    messages: list[dict] = []
    print("Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]

        reply = messages[-1].content
        if isinstance(reply, list):
            reply = "".join(
                block.get("text", "") for block in reply if isinstance(block, dict)
            )
        print(f"Agent: {reply}")


if __name__ == "__main__":
    main()


## HOMEWORK
## Use APPLE 10 K DOCUMENTS TO ANSWER THE FOLLOWING QUESTION: What is the total revenue for Apple in 2025?  - CHUNK BY PAGE AND STORE IN PINECONE VECTOR STORE
## 1. LOADING THE APPLE 10 K DOCUMENTS TO VECTOR STORE - PINE CONE (INSTALL PINECONE SKILLS + LANGCHAIN SKILLS)
## 2. CREATE A TOOL TO QUERY THE VECTOR STORE

### ADD LANGSMITH Instrumentation to the LangGraph and langchain agents to track the execution of the tasks.