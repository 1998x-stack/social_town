#!/usr/bin/env bash
set -e

STAGE=""
MOCK_LLM=""
COVERAGE=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --stage)    STAGE="$2"; shift ;;
    --mock-llm) MOCK_LLM="1" ;;
    --coverage) COVERAGE="1" ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

# Build pytest args array (safe — no eval, no word splitting)
PYTEST_ARGS=(-v)
[ -n "$COVERAGE" ] && PYTEST_ARGS+=(--cov=. --cov-report=term-missing)
[ -n "$MOCK_LLM" ] && PYTEST_ARGS+=(-k "not real_llm")

if [ -n "$STAGE" ]; then
  # Use bash glob directly — no eval needed
  TEST_DIRS=(tests/stage${STAGE}_*)
  if [ ! -e "${TEST_DIRS[0]}" ]; then
    echo "No test directory found for stage ${STAGE}"
    exit 1
  fi
  echo "▶ Running: pytest ${TEST_DIRS[*]} ${PYTEST_ARGS[*]}"
  pytest "${TEST_DIRS[@]}" "${PYTEST_ARGS[@]}"
else
  echo "▶ Running: pytest tests/ ${PYTEST_ARGS[*]}"
  pytest tests/ "${PYTEST_ARGS[@]}"
fi
