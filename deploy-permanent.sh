#!/bin/bash
# Deploy ml-coding-interview-share to GitHub Pages (burak-amd account).
set -euo pipefail

DECK="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="ml-coding-interview-questions"
GH="${GH:-/tmp/gh_2.74.2_macOS_arm64/bin/gh}"
OWNER="${GITHUB_OWNER:-burak-amd}"
PAGES_URL="https://${OWNER}.github.io/${REPO_NAME}/"

if [ ! -x "$GH" ]; then
  echo "GitHub CLI not found at $GH"
  echo "Download from: https://cli.github.com/"
  exit 1
fi

if ! "$GH" auth status >/dev/null 2>&1; then
  echo "GitHub login required (one-time). A browser window will open."
  "$GH" auth login -h github.com -p https -w -s repo,workflow,read:org
fi

ACTIVE="$("$GH" api user -q .login 2>/dev/null || true)"
if [ "$ACTIVE" != "$OWNER" ]; then
  echo "Warning: gh active account is '${ACTIVE:-unknown}', expected '${OWNER}'."
  echo "Switch with: gh auth switch -u ${OWNER}"
fi

cd "$DECK"

# Repo-local identity only (no global git config changes)
git config user.name "burak-amd"
git config user.email "burak.uzkent@amd.com"

if [ ! -d .git ]; then
  git init -b main
  git add index.html README.md .nojekyll SHARE.md deploy-permanent.sh
  git commit -m "Add ML coding interview question bank (Python MSA)"
fi

if ! "$GH" repo view "${OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
  "$GH" repo create "${OWNER}/${REPO_NAME}" --public --source=. --remote=origin --push
else
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/${OWNER}/${REPO_NAME}.git"
  git add -A
  git diff --cached --quiet || git commit -m "Update ML coding interview question bank"
  git push -u origin main
fi

if ! "$GH" api "repos/${OWNER}/${REPO_NAME}/pages" >/dev/null 2>&1; then
  "$GH" api "repos/${OWNER}/${REPO_NAME}/pages" -X POST --input - <<EOF
{"build_type":"legacy","source":{"branch":"main","path":"/"}}
EOF
else
  "$GH" api "repos/${OWNER}/${REPO_NAME}/pages" -X PUT --input - <<EOF
{"build_type":"legacy","source":{"branch":"main","path":"/"}}
EOF
fi

echo ""
echo "Deployed. GitHub Pages may take 1–2 minutes to go live."
echo "Permanent URL: ${PAGES_URL}"
