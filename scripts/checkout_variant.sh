#!/usr/bin/env bash
# Rebuild a working copy from variant.json: clone upstream, check out base_commit, git am the patches.
# Patches of a variant whose first patch "applies_on" another variant (e.g. "corrected-baseline @ bf237d7") are applied
# after that variant's own patches. The result is checked against variant_tree.
# Usage: scripts/checkout_variant.sh variants/cuboi-g1/corrected-baseline
set -euo pipefail
v="${1%/}"
dir="$(dirname "$v")"
[ -e "$v" ] && { echo "already exists: $v (move it away before rebuilding)" >&2; exit 1; }
# Chain of variants whose patches are applied in order, ending with $v.
chain=$(python3 - "$v" <<'PY'
import json, sys, os
v = sys.argv[1]; d = os.path.dirname(v); chain = []
while True:
    meta = json.load(open(v + ".json")); chain.insert(0, v)
    on = (meta.get("patches") or [{}])[0].get("applies_on") or ""
    parent = on.split(" @")[0].strip()
    if not parent or not os.path.exists(os.path.join(d, parent + ".json")):
        break
    v = os.path.join(d, parent)
print(" ".join(chain))
PY
)
read -r repo base tree < <(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["base_repo"], s["base_commit"], s.get("variant_tree", "-"))' "$v.json")
git clone -q "$repo" "$v"
git -C "$v" checkout -q -b "$(basename "$v")" "$base"
for c in $chain; do
  git -C "$v" am -q "$PWD/$c.patches"/*.patch
done
got=$(git -C "$v" rev-parse 'HEAD^{tree}')
echo "$v @ $(git -C "$v" rev-parse --short HEAD), tree $got"
if [ "$tree" != "-" ] && [ "$got" != "$tree" ]; then echo "tree differs from variant_tree $tree" >&2; exit 1; fi
