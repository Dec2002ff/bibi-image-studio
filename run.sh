#!/usr/bin/env bash
# Starts the studio in the project environment. All arguments go to the application as is.
#
# The application opens the browser itself once the server is ready to serve
# the page: the script does not know that moment, and Python with torch takes
# seconds to start. Disable it with --no-open-browser.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$root/.venv/bin/python"

if [ ! -x "$python" ]; then
    echo "The .venv environment was not found. Create it once: ./install.sh" >&2
    exit 1
fi

if ! "$python" -c 'import torch' >/dev/null 2>&1; then
    echo "The .venv environment is broken: torch cannot be imported." >&2
    echo "    rm -rf .venv && ./install.sh" >&2
    exit 1
fi

if [ "$("$python" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')" != "cuda" ]; then
    echo "CUDA is not available: generation would run on the CPU and take hours." >&2
fi

cd "$root"
# The flag goes first: --no-open-browser from the user's arguments comes after
# it and wins, argparse takes the last value.
exec "$python" -m fooocus_qwen --open-browser "$@"
