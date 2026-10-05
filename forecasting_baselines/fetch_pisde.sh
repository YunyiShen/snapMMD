#!/bin/bash
# PI-SDE (Jiang & Wan, Bioinformatics 2024) has no licence file in its repository, so its code is not redistributed here.
# This script fetches src/model.py at the commit we used and applies the one change we made (drop the import of its
# private adjoint solver, which the driver replaces by torchsde.sdeint). Run once: bash fetch_pisde.sh
set -e
cd "$(dirname "$0")"
tmp=$(mktemp -d)
git clone -q https://github.com/QiJiang-QJ/PI-SDE "$tmp/PI-SDE"
(cd "$tmp/PI-SDE" && git checkout -q c61b3220f4df85957585561bd6b22a92c43cdc3f)
mkdir -p vendor/pisde && touch vendor/pisde/__init__.py
python3 - "$tmp/PI-SDE/src/model.py" vendor/pisde/model.py <<'PY'
import sys
src = open(sys.argv[1]).read()
assert src.count("import src.sde as sde\n") == 1
src = src.replace("import src.sde as sde\n", "# (vendored) the original imports its own adjoint solver, src.sde; ForwardSDE is not used here\n")
open(sys.argv[2], "w").write("# PI-SDE src/model.py, github.com/QiJiang-QJ/PI-SDE at c61b322, one import line changed (see fetch_pisde.sh)\n" + src)
PY
rm -rf "$tmp"
echo "vendor/pisde/model.py ready"
