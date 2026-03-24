#!/usr/bin/env bash
set -e
AGENTS="${AGENTS:-10}"
DAYS="${DAYS:-3}"
python main.py --agents "$AGENTS" --days "$DAYS" "$@"
