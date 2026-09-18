CREATE TABLE recipes (
    id INTEGER PRIMARY KEY,
    cooktime INTEGER,
    name TEXT NOT NULL,
    filename TEXT
);

CREATE TABLE ingredients (
    id INTEGER PRIMARY KEY,
    description TEXT NOT NULL UNIQUE
);

CREATE TABLE recipe_ingredients (
    recipe_id INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id) ON DELETE CASCADE,
    quantity VARCHAR(255),
    PRIMARY KEY (recipe_id, ingredient_id)
);
