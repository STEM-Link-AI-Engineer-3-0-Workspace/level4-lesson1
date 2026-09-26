<div align="center">

# From a Model Call to an Agent That Searches

**STEMLink AI Engineer Bootcamp**

[![Level 4](https://img.shields.io/badge/Level-4-6f42c1?style=for-the-badge)](#)
[![Lesson 1](https://img.shields.io/badge/Lesson-1-0969da?style=for-the-badge)](#)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![marimo](https://img.shields.io/badge/notebooks-marimo-1C7ED6)](https://marimo.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white)](https://docs.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-1C3C3C)](https://docs.langchain.com/oss/python/langgraph/overview)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-vector%20store-000000)](https://www.pinecone.io/)
[![Topics](https://img.shields.io/badge/topics-LLMs%20·%20Agents%20·%20RAG-orange)](#the-notebooks)

</div>

---

## What this is

The code for Level 4, Lesson 1 of the STEMLink AI Engineer Bootcamp. The lesson
goes from a single call to a language model all the way to an agent that
decides for itself when to search a vector database.

It is a set of numbered [marimo](https://marimo.io/) notebooks. Each one covers
one concept and runs on its own, and each one exists because the previous one
hit a wall. Everything runs on LangChain and LangGraph, but each piece is taken
apart and printed before you use the convenient wrapper around it.

One question runs through the whole lesson:

> *I am growing Bg 300 paddy under irrigation in the Dry Zone. How much urea, and when?*

A bare model invents an answer. By the end, an agent answers it correctly from
Sri Lanka Department of Agriculture passages it retrieved itself.

## What's inside

```
.
├── notebooks/          # the lesson, one marimo notebook per concept (see below)
├── scripts/            # shared setup every notebook imports
│   ├── config.py
│   ├── corpus.py
│   └── weather.py
├── .env.example        # the keys you need, copy to .env
├── .python-version     # 3.12, uv fetches it for you
├── pyproject.toml      # dependencies
└── uv.lock             # exact locked versions
```

---

## Setup

### 1 · Install uv

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and
dependencies. Follow the **[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)**,
or use the one-liner:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

You do **not** need to install Python yourself. uv reads `.python-version` and
downloads CPython 3.12 if you don't have it.

### 2 · Install the dependencies

From the project folder:

```bash
uv sync
```

This creates `.venv/` and installs the exact versions pinned in `uv.lock`,
including marimo.

### 3 · Add your keys

```bash
cp .env.example .env
```

Then fill in `.env`:

| key | where from |
|---|---|
| `OPENAI_API_KEY` | STEMLink gave you this |
| `PINECONE_API_KEY` | free account at https://app.pinecone.io (no card) |
| `PINECONE_INDEX` | leave as `fieldoracle`, or add your initials if you share a Pinecone account |
| `NTFY_TOPIC` | a unique topic name you invent, used by the notify notebook |

`.env` is gitignored. Never commit it.

### 4 · Open a notebook

**Option A: activate the virtual environment (recommended)**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

marimo edit notebooks/<notebook>.py
```

Once it's activated, `python` and `marimo` point at the project's environment
for the rest of the terminal session. Run `deactivate` when you're done.

**Option B: use `uv run` without activating**

```bash
uv run marimo edit notebooks/<notebook>.py
```

Either way, marimo opens the notebook in your browser. Run the cells top to
bottom: markdown cells explain, code cells do the thing. You can also run a
notebook as a plain script with `python notebooks/<notebook>.py` (or
`uv run notebooks/<notebook>.py`).

> marimo notebooks are plain `.py` files, not Jupyter JSON, so `git diff` on
> a notebook reads like a diff on code.

---

## Core components

The three files in `scripts/` are shared setup, not lesson concepts. The first
code cell in every notebook adds `scripts/` to the import path, so
`from config import chat_model` works wherever marimo was launched from.

### `scripts/config.py`: models and keys

| name | what it is |
|---|---|
| `CHAT_MODEL_RAW` / `CHAT_MODEL` | the chat model, as the raw OpenAI name and as LangChain's `provider:model` string |
| `EMBED_MODEL` / `EMBED_DIM` | `text-embedding-3-small`, 1536 dimensions |
| `INDEX_NAME`, `NTFY_TOPIC` | read from `.env` |
| `require(name)` | exits with a readable message when an env var is missing, instead of a stack trace |
| `chat_model(**kwargs)` | the chat model every notebook uses, via `init_chat_model` on OpenAI's Responses API, which allows tools and reasoning together |
| `embeddings()` | an `OpenAIEmbeddings` instance |
| `rule(title)` | prints a section divider so script output reads like a lesson |

### `scripts/corpus.py`: the knowledge base

- `DOCS`: sixteen short, chunk-shaped passages from the Sri Lanka Department
  of Agriculture's Rice Research and Development Institute. Each has an `id`, a
  `category` (`variety`, `fertilizer`, `establishment`, `season`, `water`), a
  `source`, and the `text`.
- `QUESTIONS`: the lesson's recurring questions, including the Bg 300 urea
  question and the Bg 352 follow-up.

### `scripts/weather.py`: place name to coordinates

A small wrapper around [Open-Meteo](https://open-meteo.com), a free weather API
that needs no account or key. `geocode(place)` turns a place name into
`(latitude, longitude, matched_name)`. `FORECAST_URL` and `TIMEZONE` are
exported for the weather tools, which live inside the notebooks so you can see
what a tool actually does.

---

## The notebooks

Each notebook arrives in its own pull request. Go through them in order.

### `00_providers_and_wrappers.py`: why a framework exists at all

Asks the same model the same question three ways (OpenAI's SDK, a raw HTTP
call, and LangChain) to show what LangChain is standing on and what it saves you.

- The provider's own SDK, then the same call with no SDK at all
- Every provider shapes its API differently
- Each provider ships its own LangChain package
- `init_chat_model("provider:model")` picks the right package for you

**Needs:** `OPENAI_API_KEY` · **Uses:** `config.CHAT_MODEL_RAW`, `config.require`

### `01_chat_and_messages.py`: messages in, a message out

Everything later is built on this: you send a list of messages and get one
message back.

- `.invoke()` with a plain string, and what's on the returned `AIMessage`
- The message roles: `SystemMessage`, `HumanMessage`, `AIMessage`
- The model remembers nothing between calls
- History is just a list you keep and resend
- Putting words in its mouth with a hand-written `AIMessage`

**Needs:** `OPENAI_API_KEY` · **Uses:** `config.chat_model`

### `02_temperature_and_top_p.py`: the two sampling dials

Runs one cheap prompt many times under different settings and prints the
results side by side.

| dial | what it does | in one word |
|---|---|---|
| `temperature` | stretches or squashes the probability distribution | how random |
| `top_p` | cuts away the unlikely words | how many compete |

- A model samples its next token rather than deciding it
- Each dial alone, then both together
- The gotcha: while reasoning is on, `top_p` is rejected and even
  `temperature=0` isn't deterministic, so every model here passes
  `reasoning_effort="none"`

**Needs:** `OPENAI_API_KEY` · **Uses:** `config.chat_model`

### `03_limitations.py`: two things a model cannot do

Asks the lesson's question about the Bg 300 urea schedule with no retrieval.
Write down the **total** urea and the **week** of the last top dressing. You'll
check both again in `12`.

- **It invents.** The answer is fluent, specific, and different every run.
- **Pasting everything in doesn't fix it.** Stuffing the corpus into the
  prompt doesn't scale.

The real answer, from the RRDI table, for a three-month variety like Bg 300:

| when | urea kg/ha |
|---|---|
| basal | 55 |
| 2 weeks | 50 |
| 4 weeks | 75 |
| 6 weeks | 65 |
| **7 weeks** | 35 |
| **total** | **225** |

**Needs:** `OPENAI_API_KEY` · **Uses:** `config.chat_model`, `corpus.DOCS`

### `04_tools_and_toolnode.py`: tools, and the loop that makes an agent

A bare model has no senses, no actions, and no loop. This notebook adds them
one at a time, around a rainfall tool that calls Open-Meteo live (free, no key).

- A tool is just a function with `@tool`, and its docstring is the prompt
- `bind_tools` tells the model the tool exists and gets back `tool_calls`
- A `ToolNode` executes the call and produces a `ToolMessage`
- Feed the result back and the model can finally answer

```
REASON    the model decides it needs something   →  AIMessage.tool_calls
ACT       the ToolNode runs your function        →  ToolMessage
OBSERVE   the result goes back into the messages →  it answers, or asks again
```

That's ReAct.

**Try this:** change `get_rainfall`'s docstring to just `"Get rainfall."` and
re-run.

**Needs:** `OPENAI_API_KEY` · **Uses:** `config.chat_model`, `weather.geocode`, `weather.FORECAST_URL`, `weather.TIMEZONE`

<!-- notebook-05 -->

<!-- notebook-06 -->

### `07_agent_notify.py`: an agent with a real-world effect

Every tool so far returned information. This one makes your phone buzz, using
two tools: `get_forecast` (Open-Meteo) and `send_alert`, which pushes a
notification through [ntfy](https://ntfy.sh).

**Setup:** install the ntfy app, subscribe to a topic name you invent, and put
the same name in `.env` as `NTFY_TOPIC`. ntfy.sh is public, so keep the topic
unique and don't send anything private.

- A question that needs both tools (forecast, then notify)
- What changes when a tool has side effects
- Trying to break it

**Try this:** find a question where the agent notifies you when it shouldn't,
then fix it by editing only the docstring.

**Needs:** `OPENAI_API_KEY`, `NTFY_TOPIC` · **Uses:** `config.chat_model`, `config.NTFY_TOPIC`, `weather.geocode`, `weather.FORECAST_URL`, `weather.TIMEZONE`

<!-- notebook-08 -->

<!-- notebook-09 -->

<!-- notebook-10 -->

<!-- notebook-11 -->

<!-- notebook-12 -->

---

## Sources

Every figure in `scripts/corpus.py` comes from the Sri Lanka Department of
Agriculture, Rice Research and Development Institute: https://doa.gov.lk.
