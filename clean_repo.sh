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

echo "== placeholder git identity in setup.sh =="
grep -n 'yash@research.local\|user.email\|user.name' setup.sh || echo "  (none)"
echo "  -> remove the hardcoded 'git config user.email/name' lines; let the"
echo "     cloning user set their own identity."

echo "== stray phase_e / phase_f result dirs (never populated) =="
grep -n 'phase_e\|phase_f' setup.sh || echo "  (none in setup.sh)"

echo "== recommended .gitignore additions =="
cat <<'EOF'
  .instance_log
  *Zone.Identifier
  __pycache__/
  *.pyc
  .venv/
EOF

echo
if [ "$APPLY" = "--apply" ]; then echo "Applied deletions. Review 'git status' and commit."
else echo "Dry run. Re-run with --apply to delete sidecars/.instance_log."; fi
