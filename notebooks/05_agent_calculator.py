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
    # 05 · `create_agent` Writes the Loop For You

    In lesson 04 you ran the `ToolNode` yourself and fed the result back by
    hand — once.

    An agent is that, repeated until the model stops asking. `create_agent`
    builds it. Four deliberately trivial tools here, so the only new thing is
    the loop.
    """)
    return


@app.cell
def _():
    from langchain.agents import create_agent
    from langchain.tools import tool

    from config import chat_model

    CALLS = []

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        CALLS.append(f"add({a}, {b})")
        print(f"   >>> add({a}, {b}) = {a + b}")
        return a + b

    @tool
    def subtract(a: float, b: float) -> float:
        """Subtract b from a."""
        CALLS.append(f"subtract({a}, {b})")
        print(f"   >>> subtract({a}, {b}) = {a - b}")
        return a - b

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        CALLS.append(f"multiply({a}, {b})")
        print(f"   >>> multiply({a}, {b}) = {a * b}")
        return a * b

    @tool
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        CALLS.append(f"divide({a}, {b})")
        print(f"   >>> divide({a}, {b}) = {a / b}")
        return a / b

    agent = create_agent(
        chat_model(),
        tools=[add, subtract, multiply, divide],
        system_prompt=(
            "You are a careful calculator. You cannot do arithmetic yourself — "
            "use the tools for every single operation, including intermediate steps."
        ),
    )
    return CALLS, agent


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · An agent is a graph, and you have seen both nodes
    """)
    return


@app.cell
def _(agent):
    print(agent.get_graph().draw_ascii())
    return


@app.cell
def _(mo):
    mo.md(r"""
    - **model** — the chat model with your tools bound to it, exactly like
      lesson 04
    - **tools** — a `ToolNode` holding your four functions, exactly like lesson
      04

    The arrow from `tools` back to `model` is the loop. That cycle is the only
    thing `create_agent` added.

    ## 2 · One tool, one step
    """)
    return


@app.cell
def _(CALLS, agent):
    CALLS.clear()
    simple_result = agent.invoke({"messages": [{"role": "user", "content": "What is 847 plus 1259?"}]})
    print()
    print(simple_result["messages"][-1].text)
    print("\ntools used:", CALLS)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Now make it chain
    """)
    return


@app.cell
def _(CALLS, agent):
    CALLS.clear()
    question = (
        "A farmer has 3 plots of 1.75 hectares each. "
        "Urea is applied at 225 kg per hectare, and a bag holds 50 kg. "
        "How many bags does he need, and how much is left over?"
    )
    print("Q:", question)
    print()
    chain_result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print()
    print(chain_result["messages"][-1].text)
    print("\ntools used, in order:")
    for _i, _c in enumerate(CALLS, 1):
        print(f"   {_i}. {_c}")
    return (chain_result,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Read the trace
    """)
    return


@app.cell
def _(chain_result):
    for _m in chain_result["messages"]:
        _kind = type(_m).__name__
        if getattr(_m, "tool_calls", None):
            for _tc in _m.tool_calls:
                print(f"   {_kind:14s} asks {_tc['name']}({_tc['args']})")
        else:
            print(f"   {_kind:14s} {_m.text[:64]!r}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What to notice

    Nobody wrote a plan. There is no `if` in this notebook deciding the order of
    operations. The model worked out that it needed the area first, then the
    total kilos, then the bags — and fed each result into the next call.

    That's the loop earning its keep: it went round as many times as the
    problem needed.

    Four tools is silly though. A real calculator is one tool.

    ---
    **Next: `06_agent_sympy.py`**.
    """)
    return


if __name__ == "__main__":
    app.run()
