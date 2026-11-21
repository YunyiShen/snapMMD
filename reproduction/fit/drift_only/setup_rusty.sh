#!/usr/bin/env bash
# Isolated CPU or CUDA environment: bash setup_rusty.sh [cpu|cuda].
# Does not modify another project's venv or an existing CPU installation.
# Rusty's python/3.11.11 module is used by the recent cluster job templates.
set -euo pipefail
case "${1:-cpu}" in
    cpu) DEFAULT_VENV="$HOME/venvs/snapmmd-drift"; TORCH_INDEX=https://download.pytorch.org/whl/cpu ;;
    cuda) DEFAULT_VENV="$HOME/venvs/snapmmd-drift-cuda"; TORCH_INDEX=https://download.pytorch.org/whl/cu121 ;;
    *) echo "usage: bash setup_rusty.sh [cpu|cuda]" >&2; exit 2 ;;
esac
SNAPMMD_VENV=${SNAPMMD_VENV:-$DEFAULT_VENV}
module load "${SNAPMMD_PY_MODULE:-python/3.11.11}"
if [[ -e "$SNAPMMD_VENV" ]]; then
    echo "Environment already exists: $SNAPMMD_VENV. Verify it or choose a new SNAPMMD_VENV; refusing to overwrite." >&2
    exit 1
fi
python -m venv "$SNAPMMD_VENV"
"$SNAPMMD_VENV/bin/python" -m pip install 'torch==2.2.1' --index-url "$TORCH_INDEX"
"$SNAPMMD_VENV/bin/python" -m pip install 'numpy==1.23.5' 'scipy==1.10.1' 'torchsde==0.2.6' 'POT==0.9.7.post1' tqdm
"$SNAPMMD_VENV/bin/python" -m pip freeze > "$SNAPMMD_VENV/requirements.lock.txt"
"$SNAPMMD_VENV/bin/python" -c 'import torch, torchsde, numpy, ot; print(torch.__version__, torchsde.__version__, numpy.__version__, ot.__version__, "CUDA build:", torch.version.cuda)'
echo "Ready: $SNAPMMD_VENV (exact versions in requirements.lock.txt). Run numerical checks through Slurm."
