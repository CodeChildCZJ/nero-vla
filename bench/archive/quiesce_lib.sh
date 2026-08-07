#!/usr/bin/env bash
# quiesce_lib.sh -- sourced helper. Provides `qz_recent DIR SECS`, `qz_selftest`, `qz_wait`.
#
# ============================ READ THIS BEFORE "FIXING" A GUARD ============================
# The idiom used throughout this scripts/ dir --
#     while [ -n "$(find "$D" -newermt '-120 seconds' -print -quit 2>/dev/null)" ]; do sleep; done
# -- is **CORRECT AS WRITTEN**. Do not "fix" it. openpi-base spent an hour on 2026-08-05
# convincing itself this was a live defect and was one step from killing three armed waiters
# (ckpt_janitor, ckpt_loadtest, post_train_pipeline) to swap in a "fix" for a non-bug.
#
# THE TRAP THAT FAKES THE BUG (this is the durable lesson, not the find syntax):
#   * /usr/bin/find on BOTH 145 and 147 is **GNU findutils 4.9.0**. It accepts `-newermt
#     '-120 seconds'` and the guards work, verified in both directions on an actively-written
#     directory (waits while growing, releases once aged).
#   * BUT inside a Claude Code **tool call**, `find` is not that binary. The harness shell
#     snapshot installs a shell FUNCTION shadowing it (snapshot line 85, "Shadow find/grep with
#     embedded bfs/ugrep") which execs the claude binary as **bfs 4.1.1**. bfs REJECTS GNU
#     relative timestamps -- "Invalid timestamp", it only takes ISO 8601.
#   * That function is NOT exported (`export -f` appears nowhere in the snapshot), so a script
#     launched as `bash scripts/foo.sh` gets real GNU find. **Tool call and script see two
#     different `find` programs.**
#   * So spot-checking a script's find behaviour by pasting its find line into a tool call
#     measures the WRONG BINARY and reports a guard that is fine as catastrophically broken.
#     `type -a find` inside a tool call prints "find is a function" first -- that is the tell.
#
# Why the false alarm was so persuasive, and the one true thing in it: with `2>/dev/null`
# attached, a find that errors and a find that legitimately matches nothing produce the SAME
# EMPTY STRING, so the wait loop falls through reporting success either way. That masking is
# real (it is why the artifact was indistinguishable from a defect for an hour) even though on
# these hosts nothing is currently triggering it. See n17_post_train.sh, which waits on a PID rather than a pattern, for the
# sibling trap: process-alive checks that lie inside this same harness.
#
# WHAT THIS LIB IS FOR: new waiters, not retrofitting the working ones. It removes the
# stderr masking and refuses to run until the detector has answered BOTH ways -- a guard whose
# broken state and satisfied state are the same empty string is the zero-power shape the board
# already made a standing rule (strict=False->identity, use_percentiles no-op, ds.meta.stats).
# The ISO-8601 timestamp it feeds find is accepted by GNU *and* bfs, so a script using qz_*
# behaves identically whether it is run as a script or pasted into a tool call.
set -uo pipefail

# echo non-empty iff something under $1 was modified within the last $2 seconds.
# stderr deliberately NOT redirected: a predicate error must be loud, never an empty string.
qz_recent() { find "$1" -newermt "$(date -d "-$2 seconds" -Iseconds)" -print -quit; }

# Two-sided power control. One-sided cannot tell "quiesced" from "disarmed", and the opposite
# break (a detector that always fires) hangs the caller forever instead of rushing it.
qz_selftest() {
  local d rc=0
  d=$(mktemp -d) || { echo "[qz] FATAL mktemp failed"; return 1; }
  touch "$d/fresh"
  [ -n "$(qz_recent "$d" 120)" ] || { echo "[qz] FATAL positive control: a just-touched file was NOT seen as recent -> guard disarmed"; rc=1; }
  # Age the DIRECTORY too, not just the file: `find DIR` tests DIR itself and its mtime is fresh
  # from the create. (Caught by this very control on its first run -- in production matching the
  # dir is correct and wanted, since orbax adding/removing entries bumps it.)
  touch -d '1970-01-02 00:00:00' "$d/fresh" "$d"
  [ -z "$(qz_recent "$d" 120)" ] || { echo "[qz] FATAL negative control: a 1970 dir+file WAS seen as recent -> guard would never release"; rc=1; }
  rm -rf "$d"
  [ $rc -eq 0 ] && echo "[qz] selftest PASS (find=$(find --version 2>&1 | head -1)) -- proven live in both directions"
  return $rc
}

# Block until nothing under $1 has changed for $2 seconds. Self-tests first; hard-exits if the
# detector cannot be trusted -- never degrade to "assume quiesced".
qz_wait() {
  local dir="$1" secs="$2" label="${3:-$1}"
  qz_selftest || { echo "[qz] refusing to wait on $label with an unverified detector"; exit 90; }
  while [ -n "$(qz_recent "$dir" "$secs")" ]; do
    echo "[$(date '+%F %T')] [qz] $label still being written; waiting ${secs}s"
    sleep "$secs"
  done
  echo "[$(date '+%F %T')] [qz] $label quiesced (no change for ${secs}s)"
}
