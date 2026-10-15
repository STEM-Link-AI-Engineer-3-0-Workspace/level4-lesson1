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
    # 14 · Breakout — The Trip Desk

    Nothing to do with paddy. Your group runs the trip desk at a travel agency,
    and a customer has just sent this:

    > *Two of us want to take the train from Colombo to Ella this Saturday. We
    > need to get there on Saturday, before 6 pm. Travel in the best class we
    > can actually get, and find us somewhere to stay for one night rated at
    > least 4. Our budget is $25 each for the train plus our half of the room.
    > If more than one place fits, pick the best-rated one. What's the plan?*

    Build **one agent** that works this out on its own, from the timetable and
    hotel list below, and answers in a **structured** form your code can check.

    Everything you need, you met today: `@tool` and its description (04, 05),
    a calculator tool (05, 06), `create_agent` and a system prompt (05–12), and
    `response_format` (13).

    ## Your five jobs

    | | job |
    |---|---|
    | **TODO 1** | Turn the four functions into tools: add `@tool`, and write each one's docstring |
    | **TODO 2** | Write a `calculate` tool from scratch |
    | **TODO 3** | Write the `TripPlan` schema the answer must come back in |
    | **TODO 4** | Write the system prompt |
    | **TODO 5** | Build the agent with `create_agent`, using your tools, prompt and schema |

    ## The rules

    1. **No facts in your system prompt.** No train numbers, no hotel names, no
       prices, no digits at all. The agent has to look everything up. A check
       below flags it.
    2. **No arithmetic in the model's head.** Every sum goes through your
       `calculate` tool.
    3. **Don't change what the four functions do.** Decorate them and describe
       them; don't rewrite them.

    ## Scoring — out of 10

    The scorecard at the bottom checks the first seven points against the
    customer's message. The last three come from a **hidden round** your
    lecturer reveals near the end: three more customers, run against your agent
    **without changing any code**. Build an agent that plans trips, not one that
    has memorised this trip.
    """)
    return


@app.cell
def _():
    from typing import Literal

    import sympy
    from langchain.agents import create_agent
    from langchain.tools import tool
    from pydantic import BaseModel, Field

    from config import chat_model
    from trip_desk import check_prompt, scorecard

    return (
        BaseModel,
        Field,
        Literal,
        chat_model,
        check_prompt,
        create_agent,
        scorecard,
        sympy,
        tool,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## The data

    Made up for this exercise — don't plan a real trip with it. Read it: the
    customer's answer is in here, and so are the traps.
    """)
    return


@app.cell
def _():
    TRAINS = [
        # Colombo Fort -> Ella
        {"id": "T101", "name": "Morning Blue",    "origin": "Colombo Fort", "destination": "Ella",
         "departs": "05:55", "arrives": "15:40", "days": "daily",
         "fares_lkr": {"1st": 3200, "2nd": 1800, "3rd": 900}},
        {"id": "T103", "name": "Hill Express",    "origin": "Colombo Fort", "destination": "Ella",
         "departs": "08:30", "arrives": "18:45", "days": "daily",
         "fares_lkr": {"1st": 3500, "2nd": 2000, "3rd": 1000}},
        {"id": "T105", "name": "Night Mail",      "origin": "Colombo Fort", "destination": "Ella",
         "departs": "20:00", "arrives": "06:30 (+1 day)", "days": "daily",
         "fares_lkr": {"1st": 3000, "2nd": 1600, "3rd": 800}},
        {"id": "T107", "name": "Weekday Special", "origin": "Colombo Fort", "destination": "Ella",
         "departs": "07:00", "arrives": "16:10", "days": ["mon", "tue", "wed", "thu", "fri"],
         "fares_lkr": {"1st": 2800, "2nd": 1500, "3rd": 750}},
        # Colombo Fort -> Kandy
        {"id": "K201", "name": "Kandy Intercity", "origin": "Colombo Fort", "destination": "Kandy",
         "departs": "07:00", "arrives": "09:40", "days": "daily",
         "fares_lkr": {"1st": 1500, "2nd": 800, "3rd": 400}},
        {"id": "K203", "name": "Udarata Local",   "origin": "Colombo Fort", "destination": "Kandy",
         "departs": "10:30", "arrives": "13:10", "days": "daily",
         "fares_lkr": {"1st": 1400, "2nd": 750, "3rd": 380}},
        {"id": "K205", "name": "Office Express",  "origin": "Colombo Fort", "destination": "Kandy",
         "departs": "06:10", "arrives": "08:50", "days": ["mon", "tue", "wed", "thu", "fri"],
         "fares_lkr": {"1st": 1200, "2nd": 700, "3rd": 350}},
    ]

    # Seats left, by (train, class, day). Anything not listed has 20 seats.
    SEATS = {
        ("T101", "1st", "sat"): 0,
        ("T101", "2nd", "sat"): 12,
        ("T101", "3rd", "sat"): 40,
    }

    # Price per room per night, in LKR. Every room sleeps two.
    HOTELS = [
        {"name": "Nine Arches View",   "town": "Ella",  "rating": 4.6, "price_lkr": 9000},
        {"name": "Little Adam's Lodge", "town": "Ella",  "rating": 4.2, "price_lkr": 6600},
        {"name": "Ravana Rest",        "town": "Ella",  "rating": 3.7, "price_lkr": 4800},
        {"name": "Tea Terrace",        "town": "Ella",  "rating": 4.8, "price_lkr": 12000},
        {"name": "Lake View Kandy",    "town": "Kandy", "rating": 4.3, "price_lkr": 7500},
        {"name": "Temple Street Inn",  "town": "Kandy", "rating": 3.9, "price_lkr": 4200},
    ]

    LKR_PER_USD = 300

    # Every function below records itself here, so you can see what the agent did.
    CALLS = []
    return CALLS, HOTELS, LKR_PER_USD, SEATS, TRAINS


@app.cell
def _(mo):
    mo.md(r"""
    ## TODO 1 · Four functions, not yet tools

    They work — run them yourself if you like. But the model can't see a
    function; it can only see a **tool**, and it only knows what a tool does
    from its **docstring**.

    Add `@tool` above each one and write a docstring that tells the model what
    it's for, what goes in, what comes out, and anything it must not forget.
    (What does `find_trains` *not* tell you?)
    """)
    return


@app.cell
def _(CALLS, HOTELS, LKR_PER_USD, SEATS, TRAINS, tool):
    def find_trains(origin: str, destination: str, day: str) -> list[dict]:
        CALLS.append(("find_trains", origin, destination, day))
        print(f">>> find_trains: {origin} to {destination} on {day}")
        day = day.strip().lower()[:3]
        return [
            {k: t[k] for k in ("id", "name", "departs", "arrives", "fares_lkr")}
            for t in TRAINS
            if origin.lower() in t["origin"].lower()
            and destination.lower() in t["destination"].lower()
            and (t["days"] == "daily" or day in t["days"])
        ]

    def seats_left(train_id: str, travel_class: str, day: str) -> int:
        CALLS.append(("seats_left", train_id, travel_class, day))
        seats = SEATS.get((train_id.upper(), travel_class.lower(), day.strip().lower()[:3]), 20)
        print(f">>> seats_left: {train_id} {travel_class} class on {day} = {seats}")
        return seats

    def find_hotels(town: str, min_rating: float) -> list[dict]:
        CALLS.append(("find_hotels", town, min_rating))
        print(f">>> find_hotels: {town}, rated {min_rating} or more")
        return [h for h in HOTELS if h["town"].lower() == town.lower() and h["rating"] >= min_rating]

    def lkr_to_usd(amount_lkr: float) -> float:
        CALLS.append(("lkr_to_usd", amount_lkr))
        usd = round(amount_lkr / LKR_PER_USD, 2)
        print(f">>> lkr_to_usd: {amount_lkr} LKR = {usd} USD")
        return usd

    return find_hotels, find_trains, lkr_to_usd, seats_left


@app.cell
def _(mo):
    mo.md(r"""
    ## TODO 2 · A calculator, from scratch

    Write a tool called `calculate` that takes a sum as a string and returns
    the answer. Lesson 06 did this with SymPy (`sympify` is already imported).
    Keep the first line, `CALLS.append(...)`, so the scorecard can see it.
    """)
    return


@app.cell
def _(CALLS, sympy, tool):
    calculate = None  # TODO 2 — replace this line with your @tool def calculate(expression: str) -> str
    # first line of its body:  CALLS.append(("calculate", expression))
    return (calculate,)


@app.cell
def _(mo):
    mo.md(r"""
    ## TODO 3 · The shape of the answer

    Write `TripPlan` as a Pydantic model with **exactly these fields** — the
    scorecard reads them by name. Give every field a `Field(description=...)`:
    the model reads those too.

    | field | type | meaning |
    |---|---|---|
    | `train_id` | `str \| None` | the train to take, or `None` if no train works |
    | `travel_class` | `Literal["1st", "2nd", "3rd"] \| None` | the class to book |
    | `hotel` | `str \| None` | where to stay, or `None` if no hotel is needed or none works |
    | `cost_per_person_usd` | `float` | train fare plus their share of the room, per person, in USD |
    | `within_budget` | `bool` | does that cost fit the customer's budget? |
    | `reason` | `str` | one or two sentences on why this plan |

    Lesson 13 section 5 is worth rereading: what should the plan look like when
    **nothing** fits?
    """)
    return


@app.cell
def _(BaseModel, Field, Literal):
    class TripPlan(BaseModel):
        """TODO 3 — describe what this object is."""

        pass  # TODO 3 — the six fields from the table above

    return (TripPlan,)


@app.cell
def _(mo):
    mo.md(r"""
    ## TODO 4 · The system prompt

    Who is this agent, and how does it work a trip out? What does it check
    before it recommends a class? What does it do when nothing fits? Rule 1:
    **no facts**.
    """)
    return


@app.cell
def _(HOTELS, check_prompt):
    SYSTEM_PROMPT = (
        "Write your system prompt here."
    )

    check_prompt(SYSTEM_PROMPT, [h["name"] for h in HOTELS])
    return (SYSTEM_PROMPT,)


@app.cell
def _(mo):
    mo.md(r"""
    ## TODO 5 · Wire it together

    A model, a list of tools, a system prompt, and `response_format`. Put every
    tool the agent needs into `TOOLS`.
    """)
    return


@app.cell
def _(
    SYSTEM_PROMPT,
    TripPlan,
    calculate,
    chat_model,
    create_agent,
    find_hotels,
    find_trains,
    lkr_to_usd,
    seats_left,
):
    TOOLS = []  # TODO 5 — your tools go here

    agent = None  # TODO 5 — replace this with create_agent(...)
    return (agent,)


@app.cell
def _(CALLS, agent, mo):
    def ask(question: str):
        """Put one message to the agent, show what it did, and return its TripPlan."""
        mo.stop(agent is None, mo.md("**TODO 5** — build the agent first."))
        CALLS.clear()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        trip_plan = result.get("structured_response")

        print("tools used:", [call[0] for call in CALLS])
        if trip_plan is None:
            print("No structured_response. Did you pass response_format?")
        else:
            print(trip_plan)
        return trip_plan

    return (ask,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Run it on the customer's message
    """)
    return


@app.cell
def _(ask):
    CUSTOMER = (
        "Two of us want to take the train from Colombo to Ella this Saturday. We need to get "
        "there on Saturday, before 6 pm. Travel in the best class we can actually get, and find "
        "us somewhere to stay for one night rated at least 4. Our budget is $25 each for the "
        "train plus our half of the room. If more than one place fits, pick the best-rated one. "
        "What's the plan?"
    )
    plan = ask(CUSTOMER)
    return (plan,)


@app.cell
def _(mo):
    mo.md(r"""
    ## The scorecard

    Checks your `TripPlan` against the answer (the checks live in
    `scripts/trip_desk.py`, hashed, so you can't just read it off). Your
    lecturer's run decides. If a point is missing, read the
    `tools used` line: which check did the agent skip?
    """)
    return


@app.cell
def _(CALLS, SYSTEM_PROMPT, plan, scorecard):
    scorecard(plan, CALLS, SYSTEM_PROMPT)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Test it your own way

    One right answer proves very little. Invent customers and run them below:
    a different day, a different town, a trip that can't be done, a budget
    that's too small. Does `within_budget` tell the truth? Does it ever book a
    sold-out class?

    ## The hidden round

    Near the end, your lecturer shows three more customers. Run each with
    `ask(...)` **without changing any code**, and post the three `TripPlan`s
    in the channel.

    ## If you finish early

    - Turn this into a tool that **acts**: write `book_seats(train_id,
      travel_class, day, passengers)` that takes seats out of `SEATS`, and make
      the agent call it **only** when the customer says "book it".
    - Tiebreak: the group that gets 7/7 in the **fewest tool calls** wins.
    """)
    return


@app.cell
def _(ask):
    my_question = ""  # type your own customer here, then run this cell
    if my_question:
        ask(my_question)
    return


if __name__ == "__main__":
    app.run()
