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
    # 02 · The Two Dials: Temperature and Top P

    A model doesn't "choose" its next word. For every position it keeps a
    probability distribution over the whole vocabulary and **samples once** from
    it. Two knobs change that distribution before the sample happens:

    | dial | what it does | in one word |
    |---|---|---|
    | `temperature` | stretches or squashes the distribution | how random |
    | `top_p` | cuts away the unlikely words | how many compete |

    Both are easier to see than to explain, so this notebook runs one cheap
    prompt many times under different settings and prints the results side by
    side.

    One gotcha stands between you and these dials on this course's model
    (`gpt-5.6-luna`): it reasons by default, and while reasoning is on, `top_p`
    is rejected outright and even `temperature=0` stops being deterministic.
    Every model built below therefore passes `reasoning_effort="none"`. The last
    section proves why that's necessary.
    """)
    return


@app.cell
def _():
    import re

    from config import chat_model

    return chat_model, re


@app.cell
def _():
    # One prompt, many rolls. No right answer, short outputs, every roll fresh.
    PROMPT = "Pick a random whole number between 1 and 100. Reply with only the number."
    SAMPLES = 8
    return PROMPT, SAMPLES


@app.cell
def _(PROMPT, SAMPLES, re):
    def roll(model, label):
        """Roll the prompt SAMPLES times under one model config, print the spread.

        The model is asked for a bare number, so results are reduced to digits —
        otherwise an odd stray character could sneak into the printed output.
        """
        rolls = []
        for _ in range(SAMPLES):
            text = model.invoke(PROMPT).text.strip()
            match = re.match(r"\d+", text)
            rolls.append(match.group(0) if match else text)
        distinct = len(set(rolls))
        print(f"   {label:12s}  " + "  ".join(rolls) + f"   ->  {distinct} distinct")
        return rolls

    return (roll,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · A model samples — it does not decide

    For every next word the model assigns a probability to the whole vocabulary,
    then rolls a weighted die. That roll is the randomness you see between runs.
    The two knobs edit the die before it's rolled:

    - `temperature` reshapes the curve before the roll
    - `top_p` cuts the long tail off before the roll

    OpenAI's defaults are `temperature=1.0` and `top_p=1.0` — the curve exactly
    as trained, nothing removed, nothing squashed. Everything below starts from
    one of those and moves one knob. The prompt is bent on purpose: there's no
    right answer, so repeated rolls reveal the shape of the spread. Eight rolls
    per configuration.

    ## 2 · Temperature — how random

    `temperature=0` loads the die so hard the most likely word almost always
    wins. `temperature=1.6` flattens the curve until long shots get a real
    share.
    """)
    return


@app.cell
def _(chat_model, roll):
    cold = chat_model(reasoning_effort="none", temperature=0)
    hot = chat_model(reasoning_effort="none", temperature=1.6)

    _ = roll(cold, "temp=0")
    _ = roll(hot, "temp=1.6")
    return


@app.cell
def _(mo):
    mo.md(r"""
    `temperature=0` loads the die: the most likely next word keeps winning, so
    the rolls converge on one favourite. Nearly deterministic — run it many
    times and a long shot will occasionally still slip in, because 0 is a heavy
    bias, not an iron guarantee. Same question, same answer almost every time.

    `temperature=1.6` flattens the curve, so long shots get real odds and the
    rolls spread wider. There is no "right" setting — it's a mood dial. A lookup
    that must be exact can sit at 0; original prose usually gets turned up.

    And 0 does not mean "turn thinking off". It means "almost always pick the
    single most likely word". Same die, loaded differently.

    ## 3 · Top P — how much of the vocabulary is allowed

    `top_p` picks a cutoff: keep adding the most likely words until their
    combined probability reaches `p`, then throw the rest away.
    """)
    return


@app.cell
def _(chat_model, roll):
    tight = chat_model(reasoning_effort="none", top_p=0.05)
    loose = chat_model(reasoning_effort="none", top_p=1.0)

    _ = roll(tight, "top_p=0.05")
    _ = roll(loose, "top_p=1.0")
    return


@app.cell
def _(mo):
    mo.md(r"""
    At `top_p=0.05` the die has only the top handful of numbers on it — watch
    those rolls repeat their favourite. At `top_p=1.0` nothing is removed, so
    the die is the whole, untouched distribution.

    So temperature and top_p edit *different* things: one reshapes the curve,
    the other decides how much of it survives. Neither one sets the other.

    ## 4 · Both dials together
    """)
    return


@app.cell
def _(chat_model, roll):
    robot = chat_model(reasoning_effort="none", temperature=0, top_p=0.05)
    rebel = chat_model(reasoning_effort="none", temperature=1.6, top_p=1.0)

    _ = roll(robot, "0 + 0.05")
    _ = roll(rebel, "1.6 + 1.0")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Both extremes of the same two controls. One gives you the same answer
    forever, the other gives you a fresh answer every time. Everything
    real-world sits somewhere between; you now know which knob does what.

    ## 5 · The gotcha — why every model above has `reasoning_effort="none"`

    This course's model reasons by default. With reasoning turned on, the dials
    misbehave:

    - `top_p` is rejected outright by the API
    - `temperature=0` stops being deterministic, because the hidden reasoning
      steps keep sampling their own dice
    """)
    return


@app.cell
def _(chat_model):
    try:
        chat_model(top_p=0.05).invoke("Say hi")
        print("Surprise: top_p worked. (A newer model? Rerun and see.)")
    except Exception as e:
        print(f"Proof: top_p with reasoning on -> {type(e).__name__}:")
        print(f"       {str(e)[:100]}...")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Section 2's near-deterministic rolls and section 3's spread all came from
    `reasoning_effort="none"`. Keep that in the back of your head: these knobs
    belong to the generative part of the model, and reasoning is a second
    machine sitting in front of the answer.

    ## Try it yourself

    The shapes above are the lesson; the numbers are just today's dice. Rerun
    the notebook and they'll differ. Change `SAMPLES`, the prompt, or the two
    values and watch how the spread moves. Every dial position is a real API
    parameter:

    ```python
    init_chat_model(..., temperature=0.8, top_p=0.9)
    ```

    ---
    **Next: `03_limitations.py`** — two things a model cannot do.
    """)
    return


if __name__ == "__main__":
    app.run()
