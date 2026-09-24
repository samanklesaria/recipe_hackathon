"""Shared configuration, read from the environment at import.
The values live in .envrc; run `direnv allow` once and every script picks them up.
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
EMBED_BETA = float(_env("RECIPE_EMBED_BETA"))
DB = _env("RECIPE_DB")
