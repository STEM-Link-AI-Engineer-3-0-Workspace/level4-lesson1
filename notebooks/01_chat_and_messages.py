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
    # 01 · Talking to a Model: Messages In, a Message Out

    Before agents, before retrieval, there is one thing: you send a list of
    messages, and you get one message back.

    Everything in this course is built on that. Get comfortable with the four
    message types here and nothing later will surprise you.
    """)
    return


@app.cell
def _():
    from langchain.messages import AIMessage, HumanMessage, SystemMessage

    from config import chat_model

    model = chat_model()
    return AIMessage, HumanMessage, SystemMessage, model


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The simplest call

    You can hand `.invoke()` a plain string.
    """)
    return


@app.cell
def _(model):
    simple_response = model.invoke("Name the two paddy cultivation seasons in Sri Lanka.")

    print("what you sent    : a plain string")
    print("what came back    :", type(simple_response).__name__)
    print()
    print(simple_response.text)
    return (simple_response,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · What else is on that object

    `.text` is the words — what you usually want. But the response object
    carries more: usage metadata (tokens, which is what you pay for) and
    metadata about which model actually answered.
    """)
    return


@app.cell
def _(simple_response):
    print("response.usage_metadata:", simple_response.usage_metadata)
    print(
        "response.response_metadata['model_name']:",
        simple_response.response_metadata.get("model_name"),
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Tokens are what you pay for. That number is where cost lives.

    ## 3 · Messages have roles

    A conversation is a **list**. Each item says who spoke:

    - `SystemMessage` — standing instructions. Not a turn, a setting.
    - `HumanMessage` — what the user said.
    - `AIMessage` — what the model said back.
    """)
    return


@app.cell
def _(HumanMessage, SystemMessage, model):
    messages = [
        SystemMessage("You are a terse agricultural advisor for Sri Lankan farmers."),
        HumanMessage("What is Bg 300?"),
    ]

    roles_response = model.invoke(messages)
    print(roles_response.text)
    return messages, roles_response


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · The model remembers nothing

    Ask a follow-up with no history, and watch it fail.
    """)
    return


@app.cell
def _(HumanMessage, model):
    no_history_reply = model.invoke([HumanMessage("How long does it take to mature?")])
    print(no_history_reply.text[:300])
    return


@app.cell
def _(mo):
    mo.md(r"""
    It has no idea what "it" is. There is no conversation stored on the server —
    each call is stateless.

    ## 5 · History is just a list you keep

    You append the model's reply and the next question, and send the whole
    thing back.
    """)
    return


@app.cell
def _(HumanMessage, messages, roles_response):
    history_messages = messages + [
        roles_response,
        HumanMessage("How long does it take to mature?"),
    ]

    for _m in history_messages:
        print(f"   {type(_m).__name__:14s} {_m.text[:52]!r}")
    return (history_messages,)


@app.cell
def _(history_messages, model):
    history_reply = model.invoke(history_messages)
    print(history_reply.text[:300])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now it knows. Nothing was stored anywhere on the server — you resent the
    *whole* conversation. "Memory" in a chatbot is a list in your own code.

    That also means history costs tokens, and a long chat costs more per turn
    than a short one.

    ## 6 · You can put words in its mouth

    An `AIMessage` you wrote yourself is indistinguishable from one the model
    actually produced.
    """)
    return


@app.cell
def _(AIMessage, HumanMessage, SystemMessage, model):
    faked_messages = [
        SystemMessage("You are a terse agricultural advisor."),
        HumanMessage("What is Bg 300?"),
        AIMessage("Bg 300 is a three-month paddy variety."),  # the model never said this
        HumanMessage("And how much urea does that age class need in total?"),
    ]
    print(model.invoke(faked_messages).text[:250])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The third message is one *we* typed. The model can't tell.

    Useful for steering, and worth knowing ahead of later lessons: the history
    is yours to construct. That's how agent frameworks feed tool results back
    into the conversation.

    ---
    **Next: `02_temperature_and_top_p.py`** — the two dials that change how a
    model talks.
    """)
    return


if __name__ == "__main__":
    app.run()
