#!/usr/bin/env bash

dir="${1:-.}"

total=0

while IFS= read -r -d '' f; do
    n=$(awk 'END{print NR}' "$f")
    printf '%s %s\n' "$n" "$f"
    total=$((total + n))
done < <(find "$dir" -type f -print0)

printf 'TOTAL %s\n' "$total"