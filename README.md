# Level 4 · Lesson 1 — From a Model Call to an Agent That Searches

STEMLink AI Engineer Bootcamp.

Thirteen notebooks in `notebooks/`. Each one is a concept, each one runs on its own,
and each one exists because the previous one hit a wall.

`00` is a short one on why a framework exists at all. Everything after it is
LangChain and LangGraph — taken apart, one piece at a time, printing what came
back, before using the convenient wrapper around it.

---

## Setup

Python **3.10 or newer**, and [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
# Windows PowerShell:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, from this folder:

```bash
uv sync                            # creates .venv, installs the locked versions
cp .env.example .env               # fill in your keys
uv run marimo edit notebooks/00_providers_and_wrappers.py   # opens in your browser
```

Each notebook is a plain `.py` file — marimo notebooks, not Jupyter. `marimo
edit` opens one interactively; `uv run notebooks/<file>.py` runs it top to
bottom as a script, same as any of these used to run before. Cells are pure
Python functions under the hood, so `git diff` on a notebook reads like a
diff on code, not a JSON blob.

You do **not** need to install Python — `uv sync` reads `.python-version` and
fetches CPython 3.12 if you do not have it. `uv run` uses the project's
environment without you activating anything.

| key | where from |
|---|---|
| `OPENAI_API_KEY` | STEMLink gave you this |
| `PINECONE_API_KEY` | free account at https://app.pinecone.io — no card |
| `NTFY_TOPIC` | a name you invent, see `07` |

---

## The notebooks

All of them live in `notebooks/`, as marimo `.py` files. Open one with `uv
run marimo edit notebooks/<file>.py` and run cells top to bottom — markdown
cells carry the explanation, code cells carry the thing to run. Go in order.
**`10` creates the Pinecone index that `11` and `12` query.**

| notebook | concept | needs |
|---|---|---|
| `00_providers_and_wrappers.py` | SDK, plain HTTP, and why a wrapper exists | OpenAI |
| `01_chat_and_messages.py` | messages in, a message out; the four types; history | OpenAI |
| `02_temperature_and_top_p.py` | the two dials — temperature (how random) and top_p (how much of the vocabulary) | OpenAI |
| `03_limitations.py` | it invents, and you cannot paste everything in | OpenAI |
| `04_tools_and_toolnode.py` | **`@tool`, `bind_tools`, and a ToolNode running it** | OpenAI |
| | *its tool calls open-meteo.com for real — free, no key* | |
| `05_agent_calculator.py` | `create_agent`; four tools chained across one problem | OpenAI |
| `06_agent_sympy.py` | one tool, any maths; tool is truth, model narrates | OpenAI |
| `07_agent_notify.py` | a tool with an effect — your phone buzzes | OpenAI + ntfy |
| `08_embeddings.py` | text becomes 1536 numbers | OpenAI |
| `09_similarity_and_ann.py` | cosine vs euclidean by hand; O(N) exact search; ANN/HNSW idea via a neighbour-graph hop | — |
| `10_vector_store.py` | Pinecone through LangChain; documents, metadata, filters | both |
| `11_rag_query_generation.py` | retrieve then answer — and who writes the query | both |
| `12_agentic_rag.py` | hand search to the agent; it decides | both |

Three files in `scripts/` are not concepts, just shared setup that every
notebook imports — the first code cell in each notebook puts `scripts/` on
the import path, so `from config import chat_model` works no matter where
marimo's working directory happens to be:

- `config.py` — model names and key checks
- `corpus.py` — sixteen passages from the Department of Agriculture
- `weather.py` — turns a place name into coordinates for the tools in `04` and `07`

`09` needs no API key at all — it is local maths and a small neighbour-graph walk.

---

## The spine

One question, all the way through:

> *I am growing Bg 300 paddy under irrigation in the Dry Zone. How much urea, and when?*

1. **`03`** — invented. Fluent, specific, and different every run.
2. **`11`** — answered correctly, from passages retrieved out of Pinecone.
3. **`12`** — and the follow-up about **Bg 352**, which `11` cannot answer,
   works because the agent searches a second time.

The real answer, from the RRDI table, for a three-month variety like Bg 300:

| when | urea kg/ha |
|---|---|
| basal | 55 |
| 2 weeks | 50 |
| 4 weeks | 75 |
| 6 weeks | 65 |
| **7 weeks** | 35 |
| **total** | **225** |

Bg 352 is a 3.5-month variety, which moves that last dressing to **week 8**.
Answering it needs two passages joined together, which is why one search fails.

---

## Two axes, and they are independent

| | where the facts live | who decides to look | Bg 352 |
|---|---|---|---|
| `03` | the model's weights | nobody | invents |
| `11` | Pinecone | your code, once | fails |
| `12` | Pinecone | the agent | works |

**Where the facts live** is an infrastructure decision.
**Who decides to look** is an architecture decision.

Neither fixes the other, and most confused arguments about AI systems are two
people changing different axes.

---

## What an agent actually is

A Roomba senses the floor, decides where to go, moves, and looks again — a loop,
pointed at a goal. A bare language model has none of that. It reads text and
writes text, once.

`04` gives it the missing three, one at a time:

```
REASON    the model decides it needs something   →  AIMessage.tool_calls
ACT       the ToolNode runs your function        →  ToolMessage
OBSERVE   the result goes back into the messages →  it answers, or asks again
```

That is ReAct. `create_agent` in `05` is exactly this with the loop already
written — print `agent.get_graph().draw_ascii()` and you will see a `model` node,
a `tools` node, and an arrow from tools back to model. The arrow is the loop.

---

## Homework

Post in the channel before the next session — this is checked:

1. Your search count and the Bg 352 answer from `12`.
2. From `06`, a question where the tool's answer and the model's narrated steps
   disagree. Post both.

Then:

3. In `07`, make the agent over-send: find a question where it pushes to your
   phone when it should not. Then fix it by editing the docstring only.
4. In `04`, change `get_rainfall`'s docstring to just `"Get rainfall."` and
   re-run. The description **is** the prompt.
5. In `09`, the graph demo links each vector to its three nearest neighbours.
   Change that 3 to 1 and re-run. What happens to the hop path to the right
   cluster — and why?
6. In `10`, add five passages of your own with a new category, then check the
   filter picks them up. You are starting your FieldOracle corpus.

---

## Sources

Every figure in `corpus.py` comes from the Sri Lanka Department of Agriculture,
Rice Research and Development Institute — https://doa.gov.lk.
