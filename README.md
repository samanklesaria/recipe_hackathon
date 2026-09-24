# recipe_hackathon

A pile of cookbook epubs goes in. A week of dinners, with a shopping list sorted
by which store to buy it at, comes out.

Two halves. The first reads the books: an LLM pass over epub text that yields
recipes, ingredients, pairings and components in DuckDB. The second decides
where to shop: a chat model labels how reliably each store stocks each
ingredient, and those labels are distilled into a fine-tuned 300m embedding
model so the answer is a dot product instead of a prompt.

## Setup

`src/settings.py` reads `RECIPE_SERVER`, `RECIPE_CHAT_MODEL`,
`RECIPE_EMBED_MODEL`, `RECIPE_DB` from the environment. 
The model names are preset sections in llama.cpp router's `models.ini`, not
Hugging Face repo ids.

## The current pipeline

```bash
./scripts/alias2symlink.sh example_cookbooks/*     # macOS aliases -> symlinks, once
./scripts/book_loader.sh          # schema + every epub under example_cookbooks/
uv run python scripts/dedup_ingredients.py     # writes ingredient_merges.csv
uv run python scripts/dedup_ingredients.py --apply   # after reading the csv
uv run python scripts/ingredient_expand.py     # gloss each ingredient
uv run python scripts/load_stores.py           # store blurbs -> training_stores
uv run python scripts/generate_teaching_data.py # generate examples for embedding fine-tuning
uv run python finetune/train.py # fine-tune the embeddings
```

Everything after `book_loader.sh` is re-runnable and resumes where it stopped.
`ingredient_expand.py` and `generate_teaching_data.py` want llama-server;
`train.py` wants a GPU and does not.

To migrate and old database to a new schema, use [winged migration](https://github.com/samanklesaria/winged_migration).

**Extraction.** `src/epub_extract.py` takes recipe boundaries from the epub's own
table of contents: leaf entries mark recipes, the unit of text is the spine
document or the slice between two anchors. Books with only chapter-level
contents yield chapter-sized units and the model splits those. No per-book
parsing rules. `--limit` caps units per book, which is how you iterate on the
prompt cheaply. Front and back matter get skipped on the OPF's own say-so,
cross-references become `[[R3]]` markers the model can resolve into
`recipe_requires` and `recipe_pairings`, and an extracted ingredient is thrown
away unless its text actually appears in the source, which is the only defense
against a model inventing one.

**Dedup.** `scripts/dedup_ingredients.py` uses the corpus as its own oracle: A is
an alias of B when B's lemmas are a subsequence of A's and B is used alone often
enough to look canonical. "crispy bacon" folds into "bacon"; "olive oil" does
not fold into "oil", because "oil" never stands alone. It writes proposals to
CSV so you can read them before applying.

**Glosses.** A 300m encoder does not know what urad dal is. `ingredient_expand.py`
asks the chat model for one line — "split black lentil, South Asian pulse, sold
dried" — under a GBNF grammar that bounds the answer rather than truncating it.
The gloss is what gets embedded, in training and at query time both.

**Teaching data.** `generate_teaching_data.py` walks every (store, gloss) pair and
makes the chat model answer `always | usually | sometimes | never`, again by
grammar. Store-major order, because the store paragraph is the shared prompt
prefix llama-server keeps in its KV cache. Both scripts commit per row and skip
what they already have, so an interrupted run costs nothing twice.

**Fine-tune.** `finetune/train.py` maps those four words to probabilities and
trains a LoRA over `unsloth/embeddinggemma-300m` as a two-tower matcher, BCE on
the raw dot product. Not cosine: a well-stocked supermarket ought to be able to
say so with a larger norm, and normalizing throws that away.

## Planning a week

`recipes.sql` defines `plan_week(n)`, which is the point of all of it.

```sql
SELECT * FROM plan_week(5);
-- recipe_id | name | cooktime | is_side | for_recipe | ingredients
```

It picks `n` non-side recipes not cooked in the last two weeks, ranked by a
Thompson draw: upvotes and downvotes are the parameters of a Beta over how good
the recipe is, `beta_sample` draws from it, and the top `n` win. Something
downvoted forty times will still come up eventually. Along with each pick come
the recipes the book says to serve with it and, recursively, the sauces and
spice blends those depend on; `for_recipe` says what pulled each one in.

The Beta draw is a normal approximation, since DuckDB has no gamma variate. It
is noted in the schema.

## Tests

```bash
uv run pytest
```

## Future work
- Allow removing items from ingredients list that we already have. 
- Flag vegetarian and vegan recipes. 
- Within a grocery store, sort into produce, dairy, cans & bottles, frozen, other-refrigerated. Or we could just cluster the embeddings.
- Over-sample hard cases (nearby but different labels) when building embedding model.
- Parsing epubs needs debugging.
- The GUI should allow you to add cookbooks incrementally rather than all at once
- The GUI should allow you to pin the sampled recipes you like and re-sample the rest
- Bug: opening in Calibre jumps to the book start after a pause. 
- Should look over generated training data to catch obvious errors.
- Remove desserts, appetizers, side salads.
- Think about how this could be released more generally. 
- Make a grocery list view iOS app for use while shopping. This should sync to the PyQt GUI and among other shoppers in the family.
