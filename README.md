# week02 — LangChain tools agent, Pinecone RAG, and LangGraph agents

Homework for week 02. Everything runs on **Groq** (`openai/gpt-oss-120b`) through its
OpenAI-compatible API, with **LangSmith** tracing enabled.

## Homework: Apple 10-K → Pinecone → agent tool

> **Question: What is the total revenue for Apple in 2025?**
>
> **Answer: $416,161 million (about $416.2 billion)** — total net sales for fiscal 2025,
> up 6% from $391,035 million in 2024. Found on pages 26 and 39 of the 10-K.

The agent answers it from the 10-K in Pinecone, citing the page:

```
$ uv run main.py
You: What is the total revenue for Apple in 2025?
Agent: Apple's total revenue (net sales) for fiscal 2025 was $416.161 billion [Page 26].
```

### 1. Load the 10-K into Pinecone — [`ingest.py`](ingest.py)

Reads `_10-K-2025-As-Filed.pdf` with `pypdf`, **chunks it by page** (one page = one
chunk = one vector), and upserts all 80 pages into the Pinecone index `apple-10k`.

```bash
uv run ingest.py     # run once; re-running overwrites the same page IDs
```

| | |
|---|---|
| Index | `apple-10k` (serverless, AWS `us-east-1`) |
| Vectors | 80 — one per PDF page, IDs `page-1` … `page-80` |
| Embeddings | `llama-text-embed-v2`, hosted by Pinecone, 1024 dimensions |
| Metadata | `source` (file name), `page` (page number), page text |

Embeddings are Pinecone-hosted because Groq does not serve an embedding model, so
`PINECONE_API_KEY` is the only key needed for indexing. The model's 2,048-token limit
comfortably fits the longest page of this 10-K, so no page is truncated.

What `ingest.py` does, in order:

1. `PdfReader` → one `Document` per page, with `{"source": ..., "page": n}` metadata.
2. `pc.create_index(...)` if `apple-10k` does not exist yet, then wait until it is ready.
3. `PineconeVectorStore.add_documents(docs, ids=["page-1", ...])` — Pinecone embeds each
   page and upserts it. Stable IDs make re-runs idempotent.

Verify the index from the Pinecone console (Indexes → `apple-10k` → Browse), or:

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os; from pinecone import Pinecone
idx = Pinecone(api_key=os.environ['PINECONE_API_KEY']).Index('apple-10k')
print(idx.describe_index_stats())
print(idx.fetch(ids=['page-26']).vectors['page-26'].metadata['page'])
"
# {'dimension': 1024, 'total_vector_count': 80, ...}
# 26.0
```

### 2. A tool to query the vector store — `search_apple_10k` in [`main.py`](main.py)

```python
def search_apple_10k(query: str) -> str:
    """Search Apple's 2025 Form 10-K annual report for financial and business
    information, e.g. revenue, net sales, net income, risks or segments."""
    docs = get_vectorstore().similarity_search(query, k=4)
    return "\n\n---\n\n".join(
        f"[Page {int(d.metadata['page'])}]\n{d.page_content}" for d in docs
    )
```

The tool returns the 4 most similar pages with their page numbers, and the agent is
instructed to answer Apple financial questions only from those results and cite the page.

### Skills installed

- **LangChain skills** — `npx skills add langchain-ai/langchain-skills --agent claude-code --skill '*' --yes --global`
- **Pinecone skills** — `npx skills add pinecone-io/skills --agent claude-code --skill '*' --yes --global`

## Projects

### [`main.py`](main.py) — LangChain tool-calling agent

A conversational agent (`create_agent`) with three tools:

| Tool | What it does |
|---|---|
| `search_apple_10k` | Semantic search over the Apple 10-K in Pinecone (the homework) |
| `get_stock_price` | Last closing price for a ticker, via `yfinance` |
| `get_weather` | Current weather for a city, via WeatherAPI.com |

```bash
uv run main.py       # type 'exit' or 'quit' to end
```

### [`langgraph_branching_agent/agent.py`](langgraph_branching_agent/agent.py) — branching

A `StateGraph` that classifies a topic as **technical** or **business**, then routes to
the matching report node:

```
        classifier
        /        \
   technical    business
       |           |
  tech report   biz report
```

```bash
uv run langgraph_branching_agent/agent.py
```

### [`langgraph_branching_agent/parallel_agent.py`](langgraph_branching_agent/parallel_agent.py) — parallel + aggregator

A `StateGraph` where an **orchestrator** classifies the topic and plans three sections,
three **workers run at the same time** to write one section each, and an **aggregator**
merges them into one report:

```
        Orchestrator
       /     |      \
  Worker 1  Worker 2  Worker 3      (parallel)
       \     |      /
         Aggregator
```

The three workers are fan-out edges from `orchestrator`, so LangGraph runs them in the
same superstep; `aggregator` has an edge from each worker, so it waits for all three.
Because the workers write to the same state key, `sections` uses an
`Annotated[list, operator.add]` reducer to concatenate their results instead of
overwriting. A timed run shows the overlap — all three start together and the whole
report takes about as long as one worker:

```
worker 2: start 0.72s  end 1.44s
worker 3: start 0.72s  end 1.51s
worker 1: start 0.64s  end 1.53s
total 1.54s
```

```bash
uv run langgraph_branching_agent/parallel_agent.py
```

Both LangGraph scripts export their diagram (`graph.mmd` / `graph.png` and
`parallel_graph.mmd` / `parallel_graph.png`) each time they run.

### LangGraph Studio

All three graphs are registered in [`langgraph.json`](langgraph.json) — `tools_agent`,
`branching_agent` and `parallel_agent`:

```bash
uv run langgraph dev
```

Then open the Studio URL it prints (Chrome recommended; Safari blocks localhost).

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your API keys
uv run ingest.py       # load the 10-K into Pinecone (once)
uv run main.py
```

Environment variables (see [`.env.example`](.env.example)):

| Variable | Used for |
|---|---|
| `GROQ_API_KEY` | Chat model for all agents ([console.groq.com](https://console.groq.com/keys)) |
| `PINECONE_API_KEY` | Vector store + hosted embeddings ([app.pinecone.io](https://app.pinecone.io)) |
| `WEATHER_API_KEY` | `get_weather` tool ([weatherapi.com](https://www.weatherapi.com)) |
| `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` | Tracing ([smith.langchain.com](https://smith.langchain.com)) |

`.env` is gitignored — never commit your keys.

## LangSmith tracing

Setting `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env` traces every run of
the LangChain agent and both LangGraph agents automatically — no code changes. Traces
appear in the LangSmith project `week02-langchain-tools-agent`, showing each model call,
each tool call, and the 10-K pages Pinecone returned.
