# week02-langchain-tools-agent

Two small agent examples built with LangChain and LangGraph.

## Projects

### `main.py` — LangChain tool-calling agent

A conversational agent (`create_agent`) with two tools:

- `get_stock_price` — last closing price for a ticker via `yfinance`
- `get_weather` — current weather for a city via the OpenWeather API

Run it:

```bash
uv run main.py
```

### `langgraph_branching_agent/` — LangGraph branching agent

A LangGraph `StateGraph` that classifies a topic as **technical** or **business**,
then routes to a matching report-generation node:

```
classifier
  /      \
technical  business
  |          |
tech report  biz report
```

Run it directly:

```bash
uv run langgraph_branching_agent/agent.py
```

Or launch it in LangGraph Studio for local dev/inspection:

```bash
uv run langgraph dev
```

The graph diagram is exported to `langgraph_branching_agent/graph.mmd` (Mermaid source)
and `graph.png` each time the script runs.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your API keys
```

Required environment variables (see `.env.example`):

- `OPENAI_API_KEY`
- `OPENWEATHER_API_KEY`
