#!/bin/bash
# Replace each Mac alias file with a symlink to its original item.
set -euo pipefail

[ $# -gt 0 ] || { echo "usage: $0 alias-file..." >&2; exit 2; }

for a in "$@"; do
  abs=$(cd "$(dirname "$a")" && printf '%s/%s' "$PWD" "$(basename "$a")")
  # ponytail: Finder resolves the bookmark data for us; no alias parser here.
  target=$(osascript -e "tell application \"Finder\" to POSIX path of ((original item of (POSIX file \"$abs\" as alias)) as alias)") || {
    echo "not an alias (or original missing): $a" >&2; continue; }
  target=${target%/}
  rm "$abs"
  ln -s "$target" "$abs"
  echo "$abs -> $target"
done
