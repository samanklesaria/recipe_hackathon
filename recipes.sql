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
);

-- One-line gloss of an ingredient ("urad dal - split black lentil, South Asian
-- pulse, sold dried"), written by ingredient_expand.py. It is what the store
-- matcher embeds: a 300m encoder cannot tell what "urad dal" is on its own.
CREATE TABLE ingredient_expansions (
    ingredient_id INTEGER PRIMARY KEY REFERENCES ingredients(id),
    expansion VARCHAR NOT NULL
);

-- The store matcher's vector for an ingredient's gloss, written by plan.py the
-- first time that ingredient turns up on a list. A missing row means "not
-- embedded yet"; empty the table to force a recompute after a new fine-tune or
-- a change to RECIPE_EMBED_BETA. Its own table, not a column on ingredients:
-- DuckDB will not UPDATE a row that a foreign key still points at.
CREATE TABLE ingredient_embeddings (
    ingredient_id INTEGER PRIMARY KEY REFERENCES ingredients(id),
    embedding FLOAT[768] NOT NULL
);

-- Synthetic store descriptions for the store-matching fine-tune
CREATE SEQUENCE IF NOT EXISTS training_stores_id_seq;

CREATE TABLE training_stores (
    id INTEGER PRIMARY KEY DEFAULT nextval('training_stores_id_seq'),
    description VARCHAR NOT NULL UNIQUE
);

-- The user's actual grocery stores, one description per line, edited in the GUI
CREATE SEQUENCE IF NOT EXISTS stores_id_seq;

-- priority is the line's position in that tab: 0 first, and first wins when
-- more than one store can stock an ingredient.
CREATE TABLE stores (
    id INTEGER PRIMARY KEY DEFAULT nextval('stores_id_seq'),
    description VARCHAR NOT NULL UNIQUE,
    priority INTEGER NOT NULL,
    embedding FLOAT[768]
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

-- One draw from Beta(a, b), used for Thompson sampling over recipe goodness.
-- ponytail: normal approximation via Box-Muller, not a real Beta sampler --
-- DuckDB has no gamma variate. It is visibly wrong only when a or b is below
-- ~1 (we always pass counts+1, so never) and slightly over-confident in the
-- tails. Swap for a UDF if the ranking ever looks off.
CREATE OR REPLACE MACRO beta_sample(a, b) AS
    least(1.0, greatest(0.0,
        a / (a + b)
        + sqrt(a * b / ((a + b) * (a + b) * (a + b + 1)))
          * sqrt(-2 * ln(random())) * cos(2 * pi() * random())
    ));

-- A week's worth of dinners. Picks `n` non-side recipes not cooked in the last
-- two weeks, ranked by a Thompson draw on upvotes/downvotes, then returns them
-- plus everything the book says to serve alongside. One row per recipe;
-- for_recipe is NULL for a main pick and the main's id for a pairing.
CREATE OR REPLACE MACRO plan_week(n) AS TABLE (
    WITH RECURSIVE chosen AS (
        SELECT id
        FROM recipes
        WHERE NOT is_side
          AND (last_cooked IS NULL OR last_cooked <= current_date - INTERVAL 14 DAY)
        QUALIFY row_number() OVER (
            ORDER BY beta_sample(upvotes + 1, downvotes + 1) DESC
        ) <= n
    ),
    plan AS (
        SELECT id AS recipe_id, NULL::INTEGER AS for_recipe FROM chosen
        UNION
        -- pairings are stored in whichever direction the book stated them
        SELECT DISTINCT ON (p.b) p.b, c.id
        FROM chosen c
        JOIN (
            SELECT recipe_id AS a, paired_id AS b FROM recipe_pairings
            UNION ALL
            SELECT paired_id, recipe_id FROM recipe_pairings
        ) p ON p.a = c.id
        WHERE p.b NOT IN (SELECT id FROM chosen)
    ),
    -- sauces, spice blends and stocks the plan depends on, all the way down
    components AS (
        SELECT rr.requires_id AS recipe_id, rr.recipe_id AS for_recipe
        FROM recipe_requires rr
        JOIN plan p ON p.recipe_id = rr.recipe_id
        UNION
        SELECT rr.requires_id, rr.recipe_id
        FROM recipe_requires rr
        JOIN components c ON c.recipe_id = rr.recipe_id
    ),
    everything AS (
        -- a recipe reached more than one way is listed once, as a main if it is
        -- one; NULLS FIRST is what picks the main over the component row
        SELECT DISTINCT ON (recipe_id) recipe_id, for_recipe
        FROM (SELECT * FROM plan UNION ALL SELECT * FROM components)
        ORDER BY recipe_id, for_recipe NULLS FIRST
    )
    SELECT
        pl.recipe_id,
        r.name,
        r.cooktime,
        r.is_side,
        pl.for_recipe,
        (SELECT list(coalesce(ri.quantity || ' ', '') || i.description)
         FROM recipe_ingredients ri
         JOIN ingredients i ON i.id = ri.ingredient_id
         WHERE ri.recipe_id = pl.recipe_id) AS ingredients
    FROM everything pl
    JOIN recipes r ON r.id = pl.recipe_id
);
