#!/bin/bash
for test in tests/test_*; do
    echo "Running $test..."
    ./$test
    echo
done