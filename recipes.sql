-- DuckDB schema.
CREATE SEQUENCE IF NOT EXISTS recipes_id_seq;
CREATE SEQUENCE IF NOT EXISTS ingredients_id_seq;

CREATE TABLE recipes (
    id INTEGER PRIMARY KEY DEFAULT nextval('recipes_id_seq'),
    cooktime INTEGER,
    name VARCHAR NOT NULL,
    filename VARCHAR,
    -- Where the recipe sits inside that epub: the spine document, plus the
    -- fragment id it starts at when the book anchors its recipes
    -- ("OEBPS/Text/split_009.html#filepos78931").
    anchor VARCHAR,
    last_cooked DATE,
    upvotes INTEGER NOT NULL DEFAULT 0,
    downvotes INTEGER NOT NULL DEFAULT 0,
    -- A sauce, dressing or side dish not meant to be eaten on its own. A recipe
    -- for a protein component (roasted tofu, velveted chicken) is NOT a side.
    is_side BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE ingredients (
    id INTEGER PRIMARY KEY DEFAULT nextval('ingredients_id_seq'),
    description VARCHAR NOT NULL UNIQUE
);

CREATE TABLE recipe_ingredients (
    recipe_id INTEGER NOT NULL REFERENCES recipes(id),
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    quantity VARCHAR
    -- No (recipe_id, ingredient_id) primary key: a recipe may use the same
    -- ingredient twice with different amounts (minced in the sauce, whole on top).
    -- No ON DELETE CASCADE: DuckDB does not implement cascading deletes.
);

-- One-line gloss of an ingredient ("urad dal - split black lentil, South Asian
-- pulse, sold dried"), written by ingredient_expand.py. It is what the store
-- matcher embeds: a 300m encoder cannot tell what "urad dal" is on its own.
CREATE TABLE ingredient_expansions (
    ingredient_id INTEGER PRIMARY KEY REFERENCES ingredients(id),
    expansion VARCHAR NOT NULL
);

-- Synthetic store descriptions for the store-matching fine-tune, written in the
-- same voice as manually_tagged/grocery_store_examples.md but LLM-generated for
-- breadth (see finetune/store_matching.md). Training only -- the hand-labelled
-- eval stores stay in that markdown file and never land here, or the fine-tune
-- would be scored on stores it trained on.
CREATE SEQUENCE IF NOT EXISTS training_stores_id_seq;

CREATE TABLE training_stores (
    id INTEGER PRIMARY KEY DEFAULT nextval('training_stores_id_seq'),
    description VARCHAR NOT NULL UNIQUE
);

-- One (ingredient, store) pair with the LLM's ordinal guess at how often that
-- store stocks it: 'always' | 'usually' | 'sometimes' | 'never'. Stored as the
-- word, not the probability -- the word is what the GBNF grammar emits and the
-- word -> p mapping is a modelling choice that belongs in the training script.
CREATE TABLE embed_training_data (
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    training_store_id INTEGER NOT NULL REFERENCES training_stores(id),
    label VARCHAR NOT NULL,
    PRIMARY KEY (ingredient_id, training_store_id)
);

CREATE INDEX idx_recipe_ingredients_recipe ON recipe_ingredients(recipe_id);
CREATE INDEX idx_recipe_ingredients_ingredient ON recipe_ingredients(ingredient_id);

-- Recipe uses another recipe as a component: a sauce, a spice blend, a stock.
-- Directed: requires_id is a part of recipe_id.
CREATE TABLE recipe_requires (
    recipe_id INTEGER NOT NULL REFERENCES recipes(id),
    requires_id INTEGER NOT NULL REFERENCES recipes(id),
    PRIMARY KEY (recipe_id, requires_id)
);

-- Recipes suggested to be served together. Stored as written, one row per
-- direction the book states it; the book may mention it from either side only.
CREATE TABLE recipe_pairings (
    recipe_id INTEGER NOT NULL REFERENCES recipes(id),
    paired_id INTEGER NOT NULL REFERENCES recipes(id),
    PRIMARY KEY (recipe_id, paired_id)
);

CREATE INDEX idx_recipe_requires_requires ON recipe_requires(requires_id);
CREATE INDEX idx_recipe_pairings_paired ON recipe_pairings(paired_id);
