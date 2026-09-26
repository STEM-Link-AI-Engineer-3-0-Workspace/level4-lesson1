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
    # 00 · Why We Use a Framework At All

    Every other lesson in this course goes through LangChain. Before relying on
    it, let's see what it's standing on, and why it exists at all.

    We'll ask the same model the same question three ways: through OpenAI's own
    SDK, through a raw HTTP call with no SDK at all, and through LangChain.
    """)
    return


@app.cell
def _():
    from openai import OpenAI

    from config import CHAT_MODEL_RAW, require

    QUESTION = "Name the two paddy cultivation seasons in Sri Lanka."
    API_KEY = require("OPENAI_API_KEY")
    return API_KEY, CHAT_MODEL_RAW, OpenAI, QUESTION


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The provider's own SDK

    OpenAI publishes a Python library, so you install it and call methods.
    """)
    return


@app.cell
def _(API_KEY, CHAT_MODEL_RAW, OpenAI, QUESTION):
    client = OpenAI(api_key=API_KEY)
    sdk_response = client.chat.completions.create(
        model=CHAT_MODEL_RAW,
        messages=[{"role": "user", "content": QUESTION}],
    )

    print(sdk_response.choices[0].message.content)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```
    client.chat.completions.create(model=..., messages=[...])
    -> response.choices[0].message.content
    ```

    That shape — `.chat.completions.create(...)` in, `.choices[0].message.content`
    out — is convenient, and it is specific to OpenAI.

    ## 2 · When there is no SDK

    Same call, no library. This is all the SDK was doing underneath.
    """)
    return


@app.cell
def _(API_KEY, CHAT_MODEL_RAW, QUESTION):
    import requests

    raw = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": CHAT_MODEL_RAW, "messages": [{"role": "user", "content": QUESTION}]},
        timeout=60,
    ).json()

    print(raw["choices"][0]["message"]["content"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    One HTTPS POST with a JSON body, and JSON comes back. There's no magic
    anywhere in this stack — the SDK is a thin convenience wrapper over exactly
    this.

    That matters because plenty of providers ship no SDK at all. A local model
    server, a smaller vendor, your own company's internal endpoint: an HTTP call
    is all you get, and you write the plumbing yourself.

    ## 3 · And every provider is shaped differently

    The ones that *do* ship an SDK don't agree on anything:

    | provider | call | system prompt | answer lives at |
    |---|---|---|---|
    | OpenAI | `client.chat.completions.create(messages=[...])` | a message with `role="system"` | `.choices[0].message.content` |
    | Anthropic | `client.messages.create(messages=[...], max_tokens=...)` | a separate top-level argument, and `max_tokens` is required | `.content[0].text` — content is a **list** of blocks |
    | Gemini | `client.models.generate_content(contents=...)` | not even called "messages" | — |

    (Roughly — these move. That's part of the point.)

    Different argument names, different places for the system prompt, different
    ways to unwrap the answer. Swapping providers means rewriting every call
    site, and supporting two means writing both and a branch.

    ## 4 · Each provider ships its own LangChain package, too

    LangChain doesn't reimplement every provider's API itself. Instead, each
    provider gets a small integration package that translates LangChain's
    message shapes into that provider's real request — and back. OpenAI's is
    `langchain_openai`, and it exports a class called `ChatOpenAI`. You can
    import it directly, no string, no factory:
    """)
    return


@app.cell
def _(CHAT_MODEL_RAW, QUESTION):
    from langchain_openai import ChatOpenAI

    direct_model = ChatOpenAI(model=CHAT_MODEL_RAW)
    print(direct_model.invoke(QUESTION).content)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```
    ChatOpenAI(model=...).invoke(...)  ->  .content
    ```

    `ChatOpenAI` is not a generic shim — it's OpenAI-specific code, written and
    maintained against OpenAI's actual API, living in a package named after the
    provider. Anthropic's is `langchain_anthropic.ChatAnthropic`. Google's is
    `langchain_google_genai.ChatGoogleGenerativeAI`. Every provider LangChain
    supports "has one" of these, and you're free to reach for it directly when
    you know exactly which provider you're targeting and want its
    provider-specific options (like `use_responses_api` below) fully typed.

    ## 5 · `init_chat_model` picks the right package for you

    Importing the right class yourself means you still have to know, at every
    call site, which package goes with which provider. `init_chat_model` is a
    thin factory in front of exactly the classes from section 4 — it reads the
    `"provider:model"` string and hands back the matching one, already
    constructed.
    """)
    return


@app.cell
def _(CHAT_MODEL_RAW, QUESTION):
    from langchain.chat_models import init_chat_model

    wrapped_model = init_chat_model(f"openai:{CHAT_MODEL_RAW}")
    print(wrapped_model.invoke(QUESTION).text)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```
    init_chat_model("openai:gpt-5.4-mini").invoke(...)  ->  .text
    init_chat_model("anthropic:...")       .invoke(...)  ->  .text
    init_chat_model("google_genai:...")    .invoke(...)  ->  .text
    ```

    Same method, same return shape, whoever is behind it. The provider is just a
    string — under the hood `init_chat_model` builds the exact same `ChatOpenAI`
    you built by hand in section 4.

    That's what LangChain is: provider-specific packages that each speak the
    same shape, plus a factory that picks between them, plus the pieces built
    on top of it all — tools, agents, retrievers — which you get for free once
    everything speaks the same shape.

    What it costs you is sight of the request. When something is wrong and the
    wrapper isn't telling you why, come back to section 2 and make the call by
    hand.

    ---
    **Next: `01_chat_and_messages.py`** — from here on, everything is
    LangChain.
    """)
    return


if __name__ == "__main__":
    app.run()
