# Store matching

Given an ingredient from a parsed recipe and a set of grocery stores the user
has described, decide which store to buy it at — or say "none of these carry
it."

Both sides are open sets. Ingredients arrive from every new cookbook we parse;
stores are user-supplied free text and change over time. That rules out a
lookup table and rules out a fixed per-store classifier: we need a model that
takes an arbitrary store description it has never seen and an arbitrary
ingredient string and scores the pair. So, a bi-encoder.

## The model

Fine-tune `embeddinggemma-300m` into a two-tower scorer:

```
score(store, ingredient) = E_store(store) · E_ing(ingredient)
P(stocked)               = sigmoid(score)
```

Unnormalized dot product, not cosine. A store that carries a wide selection
should be able to express that as a larger embedding norm — "well-stocked
supermarket" should beat "corner bodega" on nearly every ingredient, and with
normalized vectors the only way to say that is to point at everything at once,
which the geometry won't allow.

One tower, two prompt prefixes. embeddinggemma is prompt-sensitive and already
trained with asymmetric query/document roles, so use that rather than feeding
both sides bare:

- ingredient → `task: search result | query: {ingredient}`
- store → `title: none | text: {store description}`

Loss is BCE against the label probabilities (§ Labels), which is the natural
pairing with a sigmoid over a raw dot product.

### Not doing

- **Matryoshka truncation.** A user has maybe ten stores; ingredients per shop
  are in the dozens. The whole score matrix is kilobytes. embeddinggemma is
  MRL-trained so truncation stays available for free if this ever becomes a
  bottleneck, which it will not.
- **Training the store tower and ingredient tower separately.** Shared weights,
  two prefixes. Half the parameters to fine-tune and nothing suggests the tasks
  need to diverge.

## Training data

Generated with the local llama.cpp router (`probes/models.ini`), GBNF-
constrained so every response parses — see `probes/FINDINGS.md`.

### Stores

Ask the LLM for ~200 store descriptions spanning the axes we expect users to
vary: ethnicity/specialty (east asian, south asian, halal, latin, italian deli,
health food), scale (corner bodega → hypermarket), density (dense urban core,
suburban strip mall, rural), and chain identity (named US chains, co-ops,
discount chains). Breadth here is the whole point — the model has to generalize
to store descriptions we did not write, and it can only do that if training
covered the space.

### Ingredients

**Pull from `SELECT description FROM ingredients`.** That is the real inference
distribution, it is free, and it has the messy real spellings ("urad dal",
"2 tbsp gochujang, or to taste") that synthetic lists won't. Top it up with
LLM-generated ingredients only for coverage gaps — categories our current
cookbooks happen not to use.

Expand each ingredient string once, offline, before embedding:

```
urad dal → urad dal — split black lentil, South Asian pulse, sold dried
```

The model cannot know what "urad dal" is from the token statistics of a 300m
encoder. Giving it a gloss is cheaper than any amount of fine-tuning and
probably buys more. Cache the expansions in a table keyed by ingredient id.

### Labels

For each ingredient/ store pairing, ask the LLM how often the store has the ingredient. Ask for a coarse ordinal,
which an LLM is actually good at, and do the mapping yourself:

| label | p |
|---|---|
| `always` — always stocked | 0.95 |
| `usually` | 0.80 |
| `sometimes` | 0.20 |
| `never` | 0.05 |

GBNF makes this exact:

```
root ::= "always" | "usually" | "sometimes" | "never"
```

## Evaluation

Every label above comes from the LLM, so a model that perfectly reproduces the
LLM's opinions scores 100% and may still be wrong about the world. The eval set
has to come from somewhere else.

**Hand-label ~150 (store, ingredient) pairs.** One afternoon. Use real stores
you know and ingredients from the actual recipe DB, and include the cases that
matter: things you'd expect at a specialty store and not a chain, and vice
versa. This is the only ground truth in the system and it is what sets the
threshold.

Three numbers, in order:

1. **Zero-shot baseline** — off-the-shelf embeddinggemma cosine, no training.
2. **Fine-tuned model** on the same held-out pairs.
3. **Threshold sweep** — precision/recall of the "none of these stores" answer
   as a function of the cutoff.

If (2) does not beat (1) by a clear margin, do not ship the fine-tune. The
baseline costs nothing and has no training pipeline to maintain.

For the threshold: false "no store carries this" is worse than a wrong store —
it tells the user to give up on a recipe they could have cooked. Pick the
cutoff for high recall on "some store has it" and accept the wrong-store
errors, which the user notices and corrects in the aisle.

## Build order

1. Ingredient expansion script: turn `urad dal → urad dal — split black lentil, South Asian pulse, sold dried`
2. Hand-label 150 eval pairs. Expand their ingredient lists
3. Zero-shot baseline against them. Record the number.
4. Ingest and deduplicate ingredients from cookbook db. 
5. Store description generation.
6. Pairing + GBNF ordinal labeling.
7. Fine-tune, evaluate against (3), and only then decide whether it ships.
