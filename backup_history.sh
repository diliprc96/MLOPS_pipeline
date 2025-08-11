# #!/bin/bash

# # Set history file and output path
# HISTFILE=~/.bash_history
# OUTDIR=logs
# mkdir -p "$OUTDIR"
# OUTFILE=$OUTDIR/terminal_history_$(date +%Y%m%d_%H%M%S).log

# history -a  # Write recent session history to HISTFILE
# cat "$HISTFILE" > "$OUTFILE"

# # Switch to dev branch, create if not exists
# if git rev-parse --verify dev >/dev/null 2>&1; then
#   git checkout dev
# else
#   git checkout -b dev
# fi

# # Stage, commit, and push the history log file
# git add "$OUTFILE"
# git commit -m "Automated backup terminal history $OUTFILE"
# git push origin dev

#!/bin/bash

# Set output path
OUTDIR=logs
mkdir -p "$OUTDIR"
OUTFILE=$OUTDIR/terminal_history_$(date +%Y%m%d_%H%M%S).log

# Export the current session's history
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
