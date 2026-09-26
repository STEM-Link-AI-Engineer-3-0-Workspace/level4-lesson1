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
    # 07 · The Agent Does Something in the Real World

    Every tool so far returned information. This one has an **effect** — it
    makes your phone buzz.

    **Setup, two minutes:**

    1. Install "ntfy" from the App Store or Play Store.
    2. In the app, subscribe to a topic. Invent a unique name — ntfy.sh is a
       public service and anyone who guesses your topic can read it, so don't
       put anything private in it.
    3. Put the same name in `.env` as `NTFY_TOPIC`.
    """)
    return


@app.cell
def _():
    import requests
    from langchain.agents import create_agent
    from langchain.tools import tool

    from config import chat_model, require
    from weather import FORECAST_URL, TIMEZONE, geocode

    NTFY_TOPIC = require("NTFY_TOPIC")
    SENT = []

    @tool
    def send_alert(message: str, title: str = "FieldOracle") -> str:
        """Send a push notification to the farmer's phone. Use this only when the
        farmer needs to act on something time-sensitive, such as applying fertilizer
        before forecast rain. Keep the message under twenty words."""
        SENT.append(message)
        print(f">>> send_alert: {message}")

        response = requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "default", "Tags": "seedling"},
            timeout=10,
        )
        response.raise_for_status()
        return f"Sent to the farmer's phone (status {response.status_code})."

    @tool
    def get_forecast(district: str) -> str:
        """Get the 48-hour rainfall forecast for a Sri Lankan district, in mm."""
        print(f">>> get_forecast: {district}")

        lat, lon, name = geocode(district)
        hourly = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "precipitation",
                "timezone": TIMEZONE,
                "forecast_hours": 48,     # the next 48 hours, starting now
            },
            timeout=20,
        ).json()["hourly"]

        return f"{name}: {sum(hourly['precipitation']):.1f} mm expected in the next 48 hours."

    agent = create_agent(
        chat_model(),
        tools=[get_forecast, send_alert],
        system_prompt=(
            "You advise Sri Lankan paddy farmers. Check the forecast before giving "
            "timing advice. If the farmer should act urgently, send them an alert."
        ),
    )
    return SENT, agent


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · A question that needs both tools
    """)
    return


@app.cell
def _(SENT, agent):
    SENT.clear()
    notify_question = (
        "I was going to apply urea to my field in Polonnaruwa tomorrow morning. "
        "Should I still do it? Let me know on my phone as well."
    )
    main_result = agent.invoke({"messages": [{"role": "user", "content": notify_question}]})
    print(main_result["messages"][-1].text)
    print("alerts sent:", len(SENT))
    if SENT:
        print("CHECK YOUR PHONE.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · What changed

    Every tool until now answered a question. This one **changed something**
    that exists outside the program.

    The code did not get more complicated — it's the same `@tool` decorator and
    the same loop. But the consequences did.

    - A tool that *reads* is wrong at worst.
    - A tool that *acts* is wrong, and then it has already happened.

    Notice what you were **not** asked before it fired. No confirmation step,
    no review. The model decided, and your code obeyed.

    A later lesson puts approval gates in front of tools like this. For now
    just hold the thought: the moment a tool has an effect, "the model decides"
    becomes a safety question and not only a design one.

    ## 3 · Try breaking it

    Ask something that plainly does not need an alert:

    > "What is the difference between Maha and Yala?"

    It should answer without sending anything. If it pushes to your phone
    anyway, the fault is in the tool **description**, not the model — go read
    what `send_alert`'s docstring promises, and tighten it.

    That's the exercise: make it over-send, then fix the docstring.
    """)
    return


@app.cell
def _(SENT, agent):
    SENT.clear()
    test_result = agent.invoke({"messages": [
        {"role": "user", "content": "What is the difference between Maha and Yala?"}
    ]})
    print(test_result["messages"][-1].text)
    print("alerts sent:", len(SENT))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    **Next: `08_embeddings.py`** — text becomes numbers.
    """)
    return


if __name__ == "__main__":
    app.run()
