#!/usr/bin/env bash
# Isolated CPU environment. Does not modify another project's venv.
# Rusty's python/3.11.11 module is used by the recent cluster job templates.
set -euo pipefail
SNAPMMD_VENV=${SNAPMMD_VENV:-$HOME/venvs/snapmmd-drift}
module load "${SNAPMMD_PY_MODULE:-python/3.11.11}"
if [[ -e "$SNAPMMD_VENV" ]]; then
    echo "Environment already exists: $SNAPMMD_VENV. Verify it or choose a new SNAPMMD_VENV; refusing to overwrite." >&2
    exit 1
fi
python -m venv "$SNAPMMD_VENV"
"$SNAPMMD_VENV/bin/python" -m pip install 'torch==2.2.1' --index-url https://download.pytorch.org/whl/cpu
"$SNAPMMD_VENV/bin/python" -m pip install 'numpy==1.23.5' 'scipy==1.10.1' 'torchsde==0.2.6' 'POT==0.9.7.post1' tqdm
"$SNAPMMD_VENV/bin/python" -m pip freeze > "$SNAPMMD_VENV/requirements.lock.txt"
"$SNAPMMD_VENV/bin/python" -c 'import torch, torchsde, numpy, ot; print(torch.__version__, torchsde.__version__, numpy.__version__, ot.__version__)'
echo "Ready: $SNAPMMD_VENV (exact versions in requirements.lock.txt). Run numerical checks through Slurm."
