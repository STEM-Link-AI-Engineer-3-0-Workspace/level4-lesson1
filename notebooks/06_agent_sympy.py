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
    # 06 · One Tool, Any Maths — and Who to Believe

    Four arithmetic tools was a toy. SymPy is a real computer algebra system:
    give it an expression as a string and it differentiates, integrates,
    solves, and simplifies **exactly** — not to fifteen decimal places, exactly.

    There's a second lesson hiding in here. SymPy has no general "show your
    working" feature, so the tool returns the **answer** and the model
    narrates the **steps**. That split is worth understanding, because it's how
    most useful agents are built: the tool is ground truth, the model is the
    explanation.

    And it means the model can narrate steps that don't match the tool's
    answer. The last section catches it doing exactly that.
    """)
    return


@app.cell
def _():
    import sympy
    from langchain.agents import create_agent
    from langchain.tools import tool

    from config import chat_model

    CALLS = []

    @tool
    def calculate(expression: str, operation: str = "evaluate", variable: str = "x") -> str:
        """Do exact symbolic mathematics on an expression written as a string.

        operation is one of:
          evaluate      - simplify or compute the expression
          differentiate - d/d(variable)
          integrate     - indefinite integral with respect to variable
          solve         - solve expression = 0 for variable

        Examples of expression: "x**3 * log(x)", "sin(x)/x", "2**10 + sqrt(144)"
        """
        CALLS.append((expression, operation))
        print(f"   >>> calculate({expression!r}, {operation!r}) ", end="")

        x = sympy.Symbol(variable)
        expr = sympy.sympify(expression)

        if operation == "differentiate":
            result = sympy.diff(expr, x)
        elif operation == "integrate":
            result = sympy.integrate(expr, x)
        elif operation == "solve":
            result = sympy.solve(expr, x)
        else:
            result = sympy.simplify(expr)

        print(f"-> {result}")
        return str(result)

    agent = create_agent(
        chat_model(),
        tools=[calculate],
        system_prompt=(
            "You are a mathematics tutor. You must use the calculate tool for every "
            "result — never compute anything in your head. After the tool answers, "
            "explain the steps a student would follow to get there. Always state the "
            "tool's answer exactly as it came back."
        ),
    )
    return CALLS, agent


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Arithmetic — same as lesson 05, one tool instead of four
    """)
    return


@app.cell
def _(CALLS, agent):
    CALLS.clear()
    arithmetic_result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 2**10 + sqrt(144)?"}]}
    )
    print(arithmetic_result["messages"][-1].text)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Calculus — which the four arithmetic tools could not touch
    """)
    return


@app.cell
def _(CALLS, agent):
    CALLS.clear()
    diff_result = agent.invoke({"messages": [
        {"role": "user", "content": "Differentiate x**3 * log(x) and explain the steps."}
    ]})
    print(diff_result["messages"][-1].text)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Integration
    """)
    return


@app.cell
def _(CALLS, agent):
    CALLS.clear()
    integral_result = agent.invoke({"messages": [
        {"role": "user", "content": "Integrate x * exp(x) with respect to x, and show me how."}
    ]})
    print(integral_result["messages"][-1].text)
    print("\ntool calls:", CALLS)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Tool is truth, model is narrator

    Notice the division of labour:

    - **SymPy** gave the exact answer. It cannot be wrong about algebra.
    - **Model** gave the explanation. It can be wrong about anything.

    This is the shape of almost every agent worth building. Put the thing that
    must be correct in a tool, and let the model do the part where being
    approximately right is fine.

    So always ask: which half of this answer am I trusting, and why?

    ## 5 · Catch it out

    Ask for something where the narration is easy to get wrong, then check the
    narrated steps against what the tool actually returned.
    """)
    return


@app.cell
def _(CALLS, agent):
    CALLS.clear()
    catch_result = agent.invoke({"messages": [
        {"role": "user", "content": "Integrate 1/(x**2 + 1) and walk me through it."}
    ]})
    print(catch_result["messages"][-1].text)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The tool's return value is printed above as the `>>>` line. Compare it to
    every figure in the explanation.

    If they match, good. If the narration wanders — and it sometimes does —
    you've just seen why "the model explained it confidently" is not evidence.

    **Homework:** find a question where they disagree, and post both.

    ---
    **Next: `07_agent_notify.py`** — a tool with a real-world effect.
    """)
    return


if __name__ == "__main__":
    app.run()
