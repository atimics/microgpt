#!/bin/bash
# Script to push the locally-resolved merge conflict branches to their remotes.
# All conflicts have been resolved locally; this script pushes those resolutions.
#
# Branches resolved:
#   1. copilot/fix-zero-division-error - merged main's tempfile download + branch's empty dataset check
#   2. copilot/improve-error-messages  - kept branch's comprehensive ValueError validation + main's roofline test
#   3. copilot/make-constants-configurable - kept branch's inference CLI flags + main's validation tests
#   4. copilot/fix-n-embedded-validation - clean merge (no conflicts)
#
# Usage: bash resolve_conflicts.sh

set -e

BRANCHES=(
    "copilot/fix-zero-division-error"
    "copilot/improve-error-messages"
    "copilot/make-constants-configurable"
    "copilot/fix-n-embedded-validation"
)

for branch in "${BRANCHES[@]}"; do
    echo "=== Pushing $branch ==="
    git checkout "$branch"
    git push -u origin "$branch"
    echo
done

echo "All resolved branches pushed."
