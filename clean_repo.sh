#!/usr/bin/env bash
# Strip cruft before pushing a clean public snapshot. Run from the repo root.
# Dry-run by default; pass --apply to actually delete/modify.
set -euo pipefail
APPLY=${1:-}

echo "== Windows 'Zone.Identifier' sidecars =="
find . -name '*Zone.Identifier' -print | sed 's/^/  /' || true
if [ "$APPLY" = "--apply" ]; then find . -name '*Zone.Identifier' -delete; fi

echo "== committed instance log =="
[ -f .instance_log ] && echo "  .instance_log" || echo "  (none)"
if [ "$APPLY" = "--apply" ] && [ -f .instance_log ]; then git rm --cached .instance_log 2>/dev/null || rm -f .instance_log; fi

echo "== live git identity in setup.sh (must be none; commented example is fine) =="
grep -n '^[[:space:]]*git config user\.\(email\|name\)' setup.sh || echo "  (none)"

echo "== stray phase_f references (no such experiment exists) =="
grep -rn 'phase_f' setup.sh experiments measurement analysis 2>/dev/null || echo "  (none)"

echo "== .gitignore coverage (all of these should already be present) =="
for pat in '.instance_log' '*Zone.Identifier' '__pycache__/' '*.py[codz]' '.venv'; do
  if grep -qxF "$pat" .gitignore; then echo "  ok   $pat"; else echo "  MISSING  $pat"; fi
done

echo
if [ "$APPLY" = "--apply" ]; then echo "Applied deletions. Review 'git status' and commit."
else echo "Dry run. Re-run with --apply to delete sidecars/.instance_log."; fi
