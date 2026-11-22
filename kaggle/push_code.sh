#!/usr/bin/env bash
# Upload this repository as a private Kaggle dataset (<user>/forknet-code) and run the validation kernel.
# Requires Kaggle credentials (KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json).
#   bash kaggle/push_code.sh <kaggle-username>
set -euo pipefail
USER_NAME=${1:?kaggle username}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$(mktemp -d)
git -C "$ROOT" ls-files --cached --others --exclude-standard -z | (cd "$ROOT" && xargs -0 cp --parents -t "$STAGE")
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "forknet-code", "id": "$USER_NAME/forknet-code", "licenses": [{"name": "other"}]}
JSON
if kaggle datasets status "$USER_NAME/forknet-code" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "update $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || date +%s)" --dir-mode tar
else
  kaggle datasets create -p "$STAGE" --dir-mode tar
fi
KDIR=$(mktemp -d)
cp "$ROOT/kaggle/forknet_validation.py" "$KDIR/"
sed "s/__USER__/$USER_NAME/g" "$ROOT/kaggle/kernel-metadata.json" > "$KDIR/kernel-metadata.json"
echo "Waiting for the dataset to be processed..." && sleep 60
kaggle kernels push -p "$KDIR"
echo "Follow with: kaggle kernels status $USER_NAME/forknet-validation; kaggle kernels output $USER_NAME/forknet-validation -p out/"
