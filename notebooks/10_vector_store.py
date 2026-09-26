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
    # 10 · A Managed Vector Store, Through LangChain

    Run this notebook once. It creates the Pinecone index that lessons 11 and
    12 both query.

    Lesson 09 showed why exact search doesn't scale, and the idea behind
    HNSW: links, layers, and a dial between fast and accurate. Pinecone runs
    one for you. LangChain wraps it so the code is the same shape whatever
    store is underneath — swap `PineconeVectorStore` for FAISS or Chroma and
    the rest of this notebook is unchanged.
    """)
    return


@app.cell
def _():
    import time

    from langchain_core.documents import Document
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    from config import EMBED_DIM, INDEX_NAME, embeddings, require
    from corpus import DOCS

    return (
        DOCS,
        Document,
        EMBED_DIM,
        INDEX_NAME,
        Pinecone,
        PineconeVectorStore,
        ServerlessSpec,
        embeddings,
        require,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Create the index

    The Pinecone client is only needed to create the index. After that,
    LangChain talks to it for you.
    """)
    return


@app.cell
def _(EMBED_DIM, INDEX_NAME, Pinecone, ServerlessSpec, require):
    pc = Pinecone(api_key=require("PINECONE_API_KEY"))

    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,          # 1536, because that is what the model returns
            metric="cosine",              # the metric you compared by hand in lesson 09
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("created index:", INDEX_NAME)
    else:
        print("index already exists:", INDEX_NAME)
    return (pc,)


@app.cell
def _(mo):
    mo.md(r"""
    Two decisions, and you understand both now:

    - **dimension** — how many numbers per vector (lesson 08)
    - **metric** — how "close" is measured (lesson 09)

    ## 2 · Documents, not strings

    A LangChain `Document` is text plus metadata. Metadata is what you filter
    on.
    """)
    return


@app.cell
def _(DOCS, Document):
    documents = [
        Document(
            page_content=d["text"],
            metadata={"source": d["source"], "category": d["category"]},
            id=d["id"],
        )
        for d in DOCS
    ]

    example = documents[4]
    print("page_content:", example.page_content)
    print("metadata:", example.metadata)
    print("id:", example.id)
    return (documents,)


@app.cell
def _(mo):
    mo.md(r"""
    The id is yours, so you can update or delete this passage by name later.

    ## 3 · Embed and store, in one call
    """)
    return


@app.cell
def _(INDEX_NAME, PineconeVectorStore, documents, embeddings):
    store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings())
    store.add_documents(documents)

    print("add_documents embedded every page_content, attached its metadata,")
    print("and upserted it into Pinecone. You did those three steps by hand in an earlier level.")
    return (store,)


@app.cell
def _(INDEX_NAME, documents, pc, time):
    # Serverless writes land asynchronously, so poll rather than guess a sleep.
    index = pc.Index(INDEX_NAME)
    for _ in range(60):
        count = index.describe_index_stats().get("total_vector_count", 0)
        if count >= len(documents):
            break
        time.sleep(1)
    print("vectors in the index:", count)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 · Search
    """)
    return


@app.cell
def _(store):
    def show(results):
        """Print (Document, score) pairs, one per line."""
        for doc, score in results:
            print(round(score, 3), doc.id, doc.metadata["category"])

    query = "How much urea for a three month variety, and when?"
    print("query:", query)
    show(store.similarity_search_with_score(query, k=5))
    return query, show


@app.cell
def _(mo):
    mo.md(r"""
    You never embedded the query yourself. `similarity_search` did it, ran the
    ANN search, and handed back `Document`s with their scores.

    ## 5 · Filter on metadata
    """)
    return


@app.cell
def _(query, show, store):
    print("everything:")
    show(store.similarity_search_with_score(query, k=3))

    print()
    print("fertilizer only:")
    show(store.similarity_search_with_score(query, k=3, filter={"category": "fertilizer"}))
    return


@app.cell
def _(mo):
    mo.md(r"""
    One keyword. This is the kind of routing requirement a real system needs —
    cultivation guides vs. policy documents vs. farm records is a metadata
    filter over one index, not three separate systems.

    ## 6 · The same store, as a retriever
    """)
    return


@app.cell
def _(query, store):
    retriever = store.as_retriever(search_kwargs={"k": 3})
    found = retriever.invoke(query)
    print("retriever.invoke(query) returned", len(found), "Documents:")
    for _doc in found:
        print(_doc.id, "-", _doc.page_content[:60])
    return


@app.cell
def _(mo):
    mo.md(r"""
    That interface matters. Anything shaped like a retriever can be dropped
    into a chain — or handed to an agent as a tool, which is exactly what
    lesson 12 does.

    **Keep this index.** Lessons 11 and 12 both read it.

    ---
    **Next: `11_rag_query_generation.py`**.
    """)
    return


if __name__ == "__main__":
    app.run()
