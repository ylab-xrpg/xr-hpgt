#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${repo_dir}"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "clang-format is required but was not found in PATH." >&2
  exit 1
fi

mapfile -d '' source_files < <(
  find include src app tests -type f \
    \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cu' \
       -o -name '*.h' -o -name '*.hpp' \) \
    -print0 | sort -z
)

case "${1:-}" in
  "")
    clang-format -style=file -i "${source_files[@]}"
    ;;
  --check)
    clang-format -style=file --dry-run --Werror "${source_files[@]}"
    ;;
  *)
    echo "Usage: $0 [--check]" >&2
    exit 1
    ;;
esac
