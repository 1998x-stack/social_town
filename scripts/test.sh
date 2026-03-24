#!/usr/bin/env bash
set -e

STAGE=""
MOCK_LLM=""
COVERAGE=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --stage) STAGE="$2"; shift ;;
    --mock-llm) MOCK_LLM="--mock-llm" ;;
    --coverage) COVERAGE="--cov=. --cov-report=term-missing" ;;
  esac
  shift
done

if [ -n "$STAGE" ]; then
  DIR="tests/stage${STAGE}_*"
else
  DIR="tests/"
fi

CMD="pytest $DIR -v $COVERAGE"
[ -n "$MOCK_LLM" ] && CMD="$CMD -k 'not real_llm'"

echo "▶ Running: $CMD"
eval $CMD
