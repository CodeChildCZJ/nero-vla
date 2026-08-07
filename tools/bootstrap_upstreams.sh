#!/usr/bin/env bash
# Populate third_party/ with the six upstream repositories this benchmark was built on,
# each pinned to the exact commit that was used, with our changes applied on top.
#
#   bash tools/bootstrap_upstreams.sh            # all six
#   bash tools/bootstrap_upstreams.sh lerobot    # just one
#
# You do NOT need this to reproduce the leaderboard. `python bench/scripts/score/score.py`
# needs numpy and nothing else. This is only for re-training or re-running predictions.
#
# Why a script and not git submodules: our changes to four of these upstreams live in
# patches/ as plain diffs, verified to apply at the pinned SHAs. Cloning upstream and
# applying a patch reproduces the exact tree we used, without anyone having to maintain
# four long-lived forks -- and it keeps working if a fork is ever deleted or rewritten.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party"
PATCHES="$ROOT/patches"
mkdir -p "$DEST"

# name | url | pinned sha | patch (or -) | extra files as "src::dest_relative_path"
REPOS=(
"openpi|https://github.com/Physical-Intelligence/openpi.git|c23745b5ad24|openpi.patch|-"
"openpi-agilex|https://github.com/agilexrobotics/openpi-agilex.git|432577c37c4f|openpi-agilex.patch|openpi-agilex.nero_policy.py::src/openpi/policies/nero_policy.py"
"Isaac-GR00T-n1d5|https://github.com/NVIDIA/Isaac-GR00T.git|4af2b622892f|Isaac-GR00T-n1d5.patch|Isaac-GR00T-n1d5.setup_env.sh::setup_env.sh"
"openvla-oft|https://github.com/moojink/openvla-oft.git|e4287e9|openvla-oft.patch|openvla-oft.finetune_nero.py::vla-scripts/finetune_nero.py openvla-oft.nero_dataset.py::prismatic/vla/datasets/nero_dataset.py"
"Isaac-GR00T|https://github.com/NVIDIA/Isaac-GR00T.git|b9955401d50c|-|-"
"lerobot|https://github.com/huggingface/lerobot.git|f66e5128ecb2|-|-"
)

want="${1:-}"
for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url sha patch extras <<<"$entry"
  [ -n "$want" ] && [ "$want" != "$name" ] && continue

  target="$DEST/$name"
  if [ -d "$target/.git" ]; then
    echo "== $name: already present at $target, skipping (delete it to re-bootstrap)"
    continue
  fi

  echo "== $name: cloning $url"
  git clone --quiet "$url" "$target"
  # Full clone, not --depth: a shallow clone often will not contain the pinned commit.
  git -C "$target" checkout --quiet "$sha"
  echo "   checked out $(git -C "$target" rev-parse --short HEAD)"

  if [ "$patch" != "-" ]; then
    # --check first: fail loudly on a bad apply rather than leave a half-patched tree.
    git -C "$target" apply --check "$PATCHES/$patch"
    git -C "$target" apply "$PATCHES/$patch"
    echo "   applied patches/$patch"
  fi

  if [ "$extras" != "-" ]; then
    for pair in $extras; do
      src="${pair%%::*}"; dst="${pair##*::}"
      mkdir -p "$target/$(dirname "$dst")"
      cp "$PATCHES/$src" "$target/$dst"
      echo "   added $dst"
    done
  fi
done

cat <<'EOF'

Done. What this did NOT do:

  * build any virtualenv. Each stack has its own dependency set and its own CUDA
    requirements; see the per-leg notes in bench/README.md. Isaac-GR00T-n1d5 ships
    setup_env.sh (installed above) as a starting point for that one.
  * download any base checkpoint or dataset. The NERO b2 dataset is not public --
    docs/DATA.md specifies the format so you can substitute your own.
EOF
