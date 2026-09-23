#!/bin/bash
duckdb recipes.db -f recipes.sql
find -L example_cookbooks -type f -name '*.epub' -exec uv run python -m epub_extract {} +
