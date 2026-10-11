import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    _repo_root = Path.cwd()
    while not (_repo_root / "pyproject.toml").exists():
        _repo_root = _repo_root.parent
    if str(_repo_root / "scripts") not in sys.path:
        sys.path.insert(0, str(_repo_root / "scripts"))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 12 · Agentic RAG — You Stop Being in the Loop

    In lesson 11 you did all of it: you wrote the query, you ran the search,
    you built the prompt. Retrieval happened exactly once because you wrote
    one line that did it.

    Here you hand the search to the agent as a tool and step out. It decides
    whether to search, what to search for, and how many times.

    Run `10_vector_store.py` first.
    """)
    return


@app.cell
def _():
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_pinecone import PineconeVectorStore

    from config import INDEX_NAME, chat_model, embeddings

    store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings())
    SEARCHES = []

    @tool
    def search_crop_guide(query: str, category: str | None = None) -> str:
        """Search the Sri Lankan Department of Agriculture paddy cultivation guide.

        Use the technical terms an agronomy document would use, not the farmer's
        phrasing. category is optional and may be one of: variety, fertilizer,
        establishment, water, season.
        """
        SEARCHES.append((query, category))
        print(f"   >>> search #{len(SEARCHES)}  query={query!r} category={category!r}")

        docs = store.similarity_search(
            query, k=3, filter={"category": category} if category else None
        )
        return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

    agent = create_agent(
        chat_model(),
        tools=[search_crop_guide],
        system_prompt=(
            "You are an advisor for Sri Lankan paddy farmers. Look up every figure "
            "before stating it, and search again if the first search did not give you "
            "everything you need. Quote exact figures and cite the source in square "
            "brackets. If the guide does not cover something, say so plainly."
        ),
    )
    return SEARCHES, agent


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The messy message from lesson 11 — no query writing by you
    """)
    return


@app.cell
def _(SEARCHES, agent):
    USER_TEXT = (
        "Hi, quick one -- I'm the guy with the plot near the tank in Polonnaruwa, "
        "put Bg 300 in about three weeks back. Wife says I should be putting "
        "something down again around now? What do you reckon"
    )
    SEARCHES.clear()
    main_result = agent.invoke({"messages": [{"role": "user", "content": USER_TEXT}]})
    print()
    print(main_result["messages"][-1].text)
    print("\nsearches:", len(SEARCHES))
    return


@app.cell
def _(mo):
    mo.md(r"""
    You wrote no query. The agent read the farmer's chat, worked out what to
    look up, and phrased the search itself — the job you did by hand in
    lesson 11.

    ## 2 · The question that beat lesson 11
    """)
    return


@app.cell
def _(SEARCHES, agent):
    SEARCHES.clear()
    followup_result = agent.invoke({"messages": [{"role": "user", "content":
        "I am growing Bg 352 under irrigation in the Dry Zone. "
        "How much urea, and when is the last top dressing?"}]})
    print()
    print(followup_result["messages"][-1].text)
    print("\nsearches:", len(SEARCHES))
    for _i, (_q, _c) in enumerate(SEARCHES, 1):
        print(f"   {_i}. {_q!r}  category={_c!r}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    If it searched twice, you just watched it realise — after reading the
    first result — that it needed a second thing. That's the capability lesson
    11 could not have at any query quality.

    The right answer is week 8. Check it against what it said.

    ## 3 · And it knows when not to search
    """)
    return


@app.cell
def _(SEARCHES, agent):
    SEARCHES.clear()
    offtopic_result = agent.invoke({"messages": [{"role": "user", "content":
        "Hi! What sort of things can you help me with?"}]})
    print()
    print(offtopic_result["messages"][-1].text[:220])
    print("\nsearches:", len(SEARCHES))
    return (offtopic_result,)


@app.cell
def _(mo):
    mo.md(r"""
    Zero. In lesson 11 your code would have embedded "hi" and queried Pinecone
    anyway, because retrieval was a line that always ran.

    ## 4 · The trace
    """)
    return


@app.cell
def _(offtopic_result):
    for _m in offtopic_result["messages"]:
        _kind = type(_m).__name__
        if getattr(_m, "tool_calls", None):
            for _tc in _m.tool_calls:
                print(f"   {_kind:14s} asks {_tc['name']}({_tc['args']})")
        else:
            print(f"   {_kind:14s} {_m.text[:60]!r}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · The whole session, in one table

    | lesson | who writes the query | how many searches | Bg 352 |
    |---|---|---|---|
    | 03 — bare model | — | none | invents |
    | 11 — RAG | you (or one model call you wrote) | exactly one | fails |
    | 12 — agentic RAG | the agent | zero, one, or many | works |

    Two different things changed across this course, and keeping them apart is
    most of the judgement in this subject:

    - **Where the facts live** — a model's weights → a vector store. An
      infrastructure decision.
    - **Who decides to look** — your code → the model. An architecture
      decision.

    Neither one fixes the other. A bigger index would never have answered Bg
    352, and an agent with nothing to search would have nothing to say.

    And the cost of the second one is predictability. RAG runs the same path
    every time and you can test it. An agent decides, and you cannot be sure
    what it will do on an input you haven't tried.

    A production system is often deliberately the RAG half. Get retrieval
    right before you let anything start making decisions.
    """)
    return


if __name__ == "__main__":
    app.run()
