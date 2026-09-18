"""Shared configuration, read from the environment at import.

The values live in .envrc; run `direnv allow` once and every script picks them
up. Deliberately no fallback defaults -- a default here would be a second place
for the server url to drift out of date, and a typo'd env name would silently
keep working against the wrong one.
"""
import os


def _env(name):
    try:
        return os.environ[name]
    except KeyError:
        raise SystemExit(f"{name} is not set. Run `direnv allow` in the project "
                         f"root, or source .envrc.") from None


SERVER = _env("RECIPE_SERVER")
CHAT_MODEL = _env("RECIPE_CHAT_MODEL")
EMBED_MODEL = _env("RECIPE_EMBED_MODEL")
DB = _env("RECIPE_DB")
