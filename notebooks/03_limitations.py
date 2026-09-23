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
    # 03 · Two Things a Model Cannot Do

    Bg 300 is a real Sri Lankan rice variety, and the Department of Agriculture
    publishes an exact urea schedule for it. There's one right answer, and it's
    a table of numbers.

    Write down two things while this runs: the **total** urea it recommends,
    and the **week** of the last top dressing. You'll check both again in
    lesson 12.
    """)
    return


@app.cell
def _():
    from langchain.messages import HumanMessage, SystemMessage

    from config import chat_model
    from corpus import DOCS

    model = chat_model()

    QUESTION = (
        "I am growing Bg 300 paddy under irrigation in the Dry Zone. "
        "Exactly how much urea should I apply, and at what times? "
        "Give me the schedule in kg per hectare."
    )

    SYSTEM = SystemMessage("You are an advisor for Sri Lankan paddy farmers.")
    return DOCS, HumanMessage, QUESTION, SYSTEM, model


@app.cell
def _(mo):
    mo.md(r"""
    ## Limitation 1 — it invents

    Ask the same question three times and compare.
    """)
    return


@app.cell
def _(HumanMessage, QUESTION, SYSTEM, model):
    for _attempt in (1, 2, 3):
        print(f"--- attempt {_attempt} " + "-" * 50)
        print(model.invoke([SYSTEM, HumanMessage(QUESTION)]).text[:520])
        print()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Same question, same settings, three different answers. All of them fluent,
    specific, and formatted like a real recommendation.

    This is not a bug. The model was trained to produce a plausible
    continuation, and nothing in that training separated *plausible* from
    *true*. It has no table to check against and no way to check one — so
    telling it to "be accurate" asks for a capability it does not have.

    A tiny transformer trained from scratch invents town names the same way a
    frontier model invents fertilizer schedules. Same machine, same failure,
    more parameters.

    ## Limitation 2 — you cannot fix it by pasting everything in

    Give it the source. That works — watch:
    """)
    return


@app.cell
def _(DOCS, HumanMessage, QUESTION, SystemMessage, model):
    facts = "\n\n".join(f"[{d['source']}]\n{d['text']}" for d in DOCS)
    grounded = model.invoke([
        SystemMessage(
            "You are an advisor for Sri Lankan paddy farmers. Answer ONLY from the "
            "passages below. Quote exact figures and cite the source in brackets.\n\n"
            + facts
        ),
        HumanMessage(QUESTION),
    ])
    print(grounded.text)
    return (grounded,)


@app.cell
def _(mo):
    mo.md(r"""
    That's the obvious fix: paste the whole reference document into the prompt.
    So why not always do that? Look at the cost.
    """)
    return


@app.cell
def _(grounded):
    print("that single grounded call used:", grounded.usage_metadata)
    return


@app.cell
def _(DOCS, grounded):
    corpus_tokens = grounded.usage_metadata["input_tokens"]
    per_passage = corpus_tokens / len(DOCS)

    print(f"{'corpus size':>14}  {'input tokens':>14}  {'per question':>14}")
    for n in (16, 1_000, 25_000, 400_000):
        tokens = int(per_passage * n)
        print(f"{n:>14,}  {tokens:>14,}  {'$' + format(tokens/1e6*0.15, '.4f'):>14}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    1. **It stops fitting.** A real FieldOracle corpus is every cultivation
       guide, policy document, and farm record the department publishes. That
       does not fit at any context length you can buy.
    2. **You pay for it on every single call.** Not once — every question,
       every user, forever. Look at the last row above.
    3. **Accuracy falls as you fill the window.** The counter-intuitive one:
       padding a prompt with material nobody asked about makes answers worse,
       not safer.

    So: it invents, and you cannot hand it everything.

    The rest of this course is one question — **how does it get the right
    facts, at the right moment, without you?**

    The answer has two halves. Tools, so it can go and get things (lessons
    04–07). Retrieval, so there's something worth getting (lessons 08–12).

    ---
    **Next: `04_tools_and_toolnode.py`** — giving the model tools.
    """)
    return


if __name__ == "__main__":
    app.run()
