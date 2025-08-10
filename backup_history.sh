#!/bin/bash

# Set history file and output path
HISTFILE=~/.bash_history
OUTFILE=logs/terminal_history_$(date +%Y%m%d_%H%M%S).log

# Export current shell session history to OUTFILE
history > "$OUTFILE"

# Switch to dev branch, create if not exists
if git rev-parse --verify dev >/dev/null 2>&1; then
  git checkout dev
else
  git checkout -b dev
fi

# Stage, commit, and push the history log file
git add "$OUTFILE"
git commit -m "Automated backup terminal history $OUTFILE"
git push origin dev
