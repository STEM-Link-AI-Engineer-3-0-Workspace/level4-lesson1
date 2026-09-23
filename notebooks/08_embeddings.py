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
    # 08 · Embeddings, Quickly

    You did this in an earlier level, so this is a refresher with the
    LangChain interface rather than a fresh explanation.

    An embedding turns text into a list of numbers, positioned so that things
    which *mean* similar things end up near each other.
    """)
    return


@app.cell
def _():
    from config import embeddings

    embedder = embeddings()
    return (embedder,)


@app.cell
def _(mo):
    mo.md("""
    ## 1 · One sentence in, numbers out
    """)
    return


@app.cell
def _(embedder):
    text = "Apply 55 kg/ha of urea as a basal dressing."
    vector = embedder.embed_query(text)

    print("input  :", repr(text))
    print("output : a list of", len(vector), "floats")
    print("first 8:", [round(v, 4) for v in vector[:8]])
    return


@app.cell
def _(mo):
    mo.md(r"""
    That list **is** the embedding. There is nothing else to it. Every sentence
    you embed with this model becomes exactly that many numbers, however long
    or short it is.

    ## 2 · Many at once
    """)
    return


@app.cell
def _(embedder):
    sentences = [
        "Apply 55 kg/ha of urea as a basal dressing.",       # 0
        "Urea top dressing at 4 weeks is 75 kg/ha.",         # 1  same subject
        "Maha season runs September to March.",              # 2  different subject
        "Fertiliser application rates for paddy.",           # 3  same subject, no shared words
    ]
    vectors = embedder.embed_documents(sentences)
    for _i, _s in enumerate(sentences):
        print(f"  [{_i}] {len(vectors[_i])} numbers  <- {_s}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    `embed_query` and `embed_documents` are the same maths. The split exists
    because some models encode a question differently from a passage.

    ## 3 · Why this is useful

    Look at sentences `[1]` and `[3]`. They share almost no words — "top
    dressing" versus "application rates", "urea" versus "fertiliser". A keyword
    search would rank them as unrelated.

    Their embeddings sit close together anyway, because the model learned what
    the words *mean* and not just what they look like.

    That's the whole reason to embed anything: search that survives the user
    not knowing your vocabulary.

    ---
    **Next: `09_similarity_and_ann.py`** — how "close together" is actually
    measured, and how you search millions of vectors without checking every
    one.
    """)
    return


if __name__ == "__main__":
    app.run()
