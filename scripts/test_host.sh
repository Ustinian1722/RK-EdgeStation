#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cxx="${CXX:-g++}"
"$cxx" -std=c++17 -Wall -Wextra -Wpedantic -pthread -Isrc \
  tests/cpp_core_tests.cpp src/tracker.cpp src/roi_events.cpp src/ppe_events.cpp \
  -o "$tmp/edge_core_tests"
"$tmp/edge_core_tests"
"$cxx" -std=c++17 -Wall -Wextra -Wpedantic -pthread -Isrc \
  tests/multistream_tests.cpp src/multistream.cpp -o "$tmp/multistream_tests"
"$tmp/multistream_tests"
PYTHONPATH=python python3 -m unittest discover -s tests/python -v
