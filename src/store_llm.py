import settings
import numpy as np
import requests

# Embedding Model
ING_PREFIX = "task: search result | query: "
STORE_PREFIX = "title: none | text: "

def embed(texts):
    r = requests.post(f"{settings.SERVER}/v1/embeddings", timeout=600,
                      json={"model": settings.EMBED_MODEL, "input": texts})
    r.raise_for_status()
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    return np.array([d["embedding"] for d in data])

def logits_from_embedding(glosses, stores):
    return embed([ING_PREFIX + g for g in glosses]) @ \
        embed([STORE_PREFIX + s for s in stores]).T

# LLM (distilled into embedding model)
GRAMMAR = r"""
root ::= ("always" | "usually" | "sometimes" | "never")
"""

PROMPT = """A grocery store is described below. Answer with exactly one of: \
always, usually, sometimes, never -- how reliably does this store stock the \
ingredient?

Store: {store}
Ingredient: {ingredient}
Answer:"""

def label_store(store, glosses):
    """Yield one label per gloss, all against the same store.

    Batched by store, not by ingredient, because the store paragraph sits ahead
    of the ingredient in PROMPT: every request in the batch shares that whole
    prefix, so llama-server keeps its KV cache and only prefills the two or
    three tokens of the gloss. Ingredient-major would re-prefill the store
    description every call.
    """
    for gloss in glosses:
        r = requests.post(f"{settings.SERVER}/v1/chat/completions", timeout=600, json={
            "model": settings.CHAT_MODEL,
            "messages": [{"role": "user",
                          "content": PROMPT.format(store=store, ingredient=gloss)}],
            "grammar": GRAMMAR,
            "temperature": 0,
            "max_tokens": 8,
            "cache_prompt": True,
        })
        # Not raise_for_status: llama-server explains a rejected grammar or an
        # overlong prompt in the body, and the HTTPError message drops it.
        if not r.ok:
            raise RuntimeError(f"{r.status_code} from {r.url}: {r.text[:500]}")
        yield r.json()["choices"][0]["message"]["content"].strip()
