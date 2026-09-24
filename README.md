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

<!-- notebook-00 -->

<!-- notebook-01 -->

<!-- notebook-02 -->

<!-- notebook-03 -->

<!-- notebook-04 -->

### `05_agent_calculator.py`: `create_agent` writes the loop for you

In `04` you ran the loop by hand, once. `create_agent` repeats it until the
model stops asking. Four deliberately trivial arithmetic tools keep the focus
on the loop.

- An agent is a graph: print `agent.get_graph().draw_ascii()` to see the
  `model` node, the `tools` node, and the arrow back (that arrow is the loop)
- One tool, one step, then four tools chained across one problem
- Reading the message trace

**Needs:** `OPENAI_API_KEY` · **Uses:** `config.chat_model`

<!-- notebook-06 -->

<!-- notebook-07 -->

<!-- notebook-08 -->

<!-- notebook-09 -->

<!-- notebook-10 -->

<!-- notebook-11 -->

<!-- notebook-12 -->

---

## Sources

Every figure in `scripts/corpus.py` comes from the Sri Lanka Department of
Agriculture, Rice Research and Development Institute: https://doa.gov.lk.
