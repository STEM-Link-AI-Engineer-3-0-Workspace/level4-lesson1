"""Small helpers that more than one notebook uses. Not lesson concepts.

Each one is something you have already seen written out in full in an earlier
notebook, so later notebooks can stay short.
"""

from config import INDEX_NAME, embeddings


def vector_store():
    """The Pinecone index that 10 creates, and 11 and 12 search."""
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings())


def format_docs(docs) -> str:
    """Join retrieved Documents into one block of text, each tagged with its source."""
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


def print_trace(messages) -> None:
    """Print an agent's messages one line each: the trace you read by hand in 05."""
    for m in messages:
        kind = type(m).__name__
        if getattr(m, "tool_calls", None):
            for call in m.tool_calls:
                print(f"   {kind:14s} asks {call['name']}({call['args']})")
        else:
            print(f"   {kind:14s} {m.text[:64]!r}")
