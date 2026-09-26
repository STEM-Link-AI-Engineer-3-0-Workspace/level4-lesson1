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
    # 13 · Structured Output — An Answer Your Code Can Use

    Every answer so far has been a paragraph. A person can read a paragraph; a
    program can't. To check that the last dressing is week 8, add up the urea,
    or put the schedule in a table, you'd have to pull numbers out of prose with
    string matching — and it breaks the first time the model words it
    differently.

    Structured output asks the model to answer in a **shape you define**, and
    hands you back a Python object instead of text.
    """)
    return


@app.cell
def _():
    import json
    import re
    from typing import Literal

    import sympy
    from langchain.agents import create_agent
    from langchain.messages import HumanMessage, SystemMessage
    from langchain.tools import tool
    from pydantic import BaseModel, Field

    from config import chat_model
    from corpus import DOCS
    from helpers import print_trace

    model = chat_model()

    def passages(*ids):
        """The corpus passages with these ids, joined into one block of text."""
        return "\n\n".join(d["text"] for d in DOCS if d["id"] in ids)

    GROUNDED = SystemMessage("Answer only from the passages provided.")
    return (
        BaseModel,
        Field,
        GROUNDED,
        HumanMessage,
        Literal,
        create_agent,
        json,
        model,
        passages,
        print_trace,
        re,
        sympy,
        tool,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The problem: a paragraph
    """)
    return


@app.cell
def _(GROUNDED, HumanMessage, model, passages):
    BG352 = passages("var-3.5month", "fert-3.5month")
    QUESTION = "What is the urea schedule for Bg 352 under irrigation in the Dry Zone?"

    prose = model.invoke([GROUNDED, HumanMessage(f"Passages:\n{BG352}\n\nQuestion: {QUESTION}")]).text
    print(prose)
    return BG352, QUESTION, prose


@app.cell
def _(mo):
    mo.md(r"""
    Now get "the week of the last dressing" out of that, as a number your code
    can use.
    """)
    return


@app.cell
def _(prose, re):
    print("every number in that answer:", re.findall(r"\d+(?:\.\d+)?", prose))
    return


@app.cell
def _(mo):
    mo.md(r"""
    Which one is the week? You can't tell without reading the sentence — and
    neither can your code. Run it again and the wording changes, so whatever
    pattern you write today breaks tomorrow.

    ## 2 · Describe the shape, get an object back

    A **schema** is a class that says what the answer must contain. We write it
    with Pydantic: each field has a name, a type, and a description.
    """)
    return


@app.cell
def _(BaseModel, Field, Literal):
    class Dressing(BaseModel):
        """One application of urea."""

        week: int = Field(description="Weeks after establishment. Use 0 for the basal dressing.")
        urea_kg_per_ha: float = Field(description="Urea applied at this point, in kg per hectare.")

    class Schedule(BaseModel):
        """A urea schedule for one rice variety, taken from the passages."""

        variety: str = Field(description="The variety asked about, for example 'Bg 352'.")
        age_class: Literal["3 month", "3.5 month", "4 month"] = Field(
            description="The variety's maturity age class."
        )
        dressings: list[Dressing] = Field(description="Every urea application, in order.")
        total_urea_kg_per_ha: float = Field(description="Total urea across all dressings, in kg per hectare.")

    return Dressing, Schedule


@app.cell
def _(BG352, GROUNDED, HumanMessage, QUESTION, Schedule, model):
    structured_model = model.with_structured_output(Schedule)
    schedule = structured_model.invoke([GROUNDED, HumanMessage(f"Passages:\n{BG352}\n\nQuestion: {QUESTION}")])

    print("type:", type(schedule).__name__)
    print(schedule)
    return (schedule,)


@app.cell
def _(mo):
    mo.md(r"""
    That's a `Schedule`, not a string. Every field is already the right type,
    so you just use it:
    """)
    return


@app.cell
def _(schedule):
    last = schedule.dressings[-1]
    print("variety:", schedule.variety)
    print("age class:", schedule.age_class)
    print("last dressing: week", last.week, "->", last.urea_kg_per_ha, "kg/ha")
    print()
    for _d in schedule.dressings:
        print("week", _d.week, "->", _d.urea_kg_per_ha, "kg/ha")
    return


@app.cell
def _(mo):
    mo.md(r"""
    `with_structured_output` is the same model underneath. It sends your schema
    along with the question, and parses the reply back into the class for you.

    ## 3 · Now you can test it
    """)
    return


@app.cell
def _(schedule):
    added_up = sum(_d.urea_kg_per_ha for _d in schedule.dressings)
    print("sum of the dressings:", added_up)
    print("total it reported:", schedule.total_urea_kg_per_ha)
    print("consistent?", added_up == schedule.total_urea_kg_per_ha)
    return


@app.cell
def _(mo):
    mo.md(r"""
    One line of Python checks the model against itself. With a paragraph you'd
    be reading it by eye. This is what makes a model's answer something you can
    **write a test for** — and a test is what lets you change a prompt without
    wondering what you broke.

    ## 4 · What the model was actually sent
    """)
    return


@app.cell
def _(Schedule, json):
    print(json.dumps(Schedule.model_json_schema(), indent=2))
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your class became a **JSON Schema**, and that is what travels to the model.

    - The class docstring and every field `description` go with it. They are
      prompts, exactly like a tool's docstring in lesson 04.
    - `Literal["3 month", "3.5 month", "4 month"]` means the model may only pick
      one of those three strings. Nothing else is a legal answer.
    - `list[Dressing]` nests one schema inside another.

    ## 5 · Design for the answer you might not have

    Bg 403 is a 4-month variety, and the guide has no 4-month schedule. The
    `Schedule` class above *requires* a list of dressings — so if you asked it
    about Bg 403, the model would have to produce one. A forced answer is an
    invented answer.

    Give it a legal way to say "not found":
    """)
    return


@app.cell
def _(BaseModel, Dressing, Field):
    class ScheduleLookup(BaseModel):
        """The answer to 'what is the urea schedule for this variety?', from the passages only."""

        variety: str = Field(description="The variety asked about.")
        found: bool = Field(
            description="True only if the passages give a schedule for this variety's age class."
        )
        dressings: list[Dressing] = Field(
            description="Every urea application, in order. Empty if found is false."
        )

    return (ScheduleLookup,)


@app.cell
def _(GROUNDED, HumanMessage, ScheduleLookup, model, passages):
    ALL_SCHEDULES = passages("var-3month", "var-3.5month", "var-4month", "fert-3month", "fert-3.5month")
    lookup_model = model.with_structured_output(ScheduleLookup)

    for _variety in ("Bg 300", "Bg 403"):
        _answer = lookup_model.invoke([
            GROUNDED,
            HumanMessage(f"Passages:\n{ALL_SCHEDULES}\n\nQuestion: What is the urea schedule for {_variety}?"),
        ])
        print(_variety, "found:", _answer.found, " dressings:", len(_answer.dressings))
    return


@app.cell
def _(mo):
    mo.md(r"""
    `found=False` and an empty list for Bg 403. The schema didn't just shape the
    answer — it gave the model permission to say no. When you design a schema,
    design it for the case where the answer doesn't exist.

    ## 6 · An agent that answers in a shape

    `create_agent` takes the same kind of class as `response_format`. The loop
    runs exactly as before — tool calls, observations, round again — but the
    final answer arrives as an object in `result["structured_response"]`.
    """)
    return


@app.cell
def _(BaseModel, Field, create_agent, model, sympy, tool):
    @tool
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression exactly and return the result.

        Supports + - * / ** and parentheses, plus ceiling(x) to round up.
        """
        value = sympy.sympify(expression)
        answer = str(int(value)) if value.is_Integer else f"{float(value):g}"
        print(f">>> calculate: {expression} = {answer}")
        return answer

    class BagOrder(BaseModel):
        """How much urea to buy for one field."""

        total_kg: float = Field(description="Urea needed for the whole field, in kg.")
        bags: int = Field(description="Whole bags to buy, rounded up.")
        leftover_kg: float = Field(description="Urea left over once the job is done, in kg.")

    agent = create_agent(
        model,
        tools=[calculate],
        response_format=BagOrder,
        system_prompt="You help farmers buy fertilizer. Use the calculate tool for every sum.",
    )
    return (agent,)


@app.cell
def _(agent):
    result = agent.invoke({"messages": [{"role": "user", "content":
        "I need 100 kg of urea per hectare on 3.2 hectares, and it comes in 50 kg bags. "
        "How many bags should I buy?"}]})

    order = result["structured_response"]
    print(order)
    print("bags to buy:", order.bags)
    return (result,)


@app.cell
def _(mo):
    mo.md(r"""
    The trace is still there, in `result["messages"]`, exactly like lesson 05.
    The structured answer sits beside it, under its own key.
    """)
    return


@app.cell
def _(print_trace, result):
    print_trace(result["messages"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What to take away

    | you want | you write | you get back |
    |---|---|---|
    | one model call, as data | `model.with_structured_output(Schema)` | a `Schema` object |
    | an agent's answer, as data | `create_agent(..., response_format=Schema)` | `result["structured_response"]` |

    - Field descriptions and the class docstring are prompts.
    - `Literal` limits what counts as a legal answer.
    - Give the schema a way to say "not found", or it will invent.
    - Once the answer is an object, you can test it.

    ---
    **Next: `14_breakout_trip_desk.py`** — build an agent of your own.
    """)
    return


if __name__ == "__main__":
    app.run()
