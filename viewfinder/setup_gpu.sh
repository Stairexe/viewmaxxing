#!/usr/bin/env bash
# One-time setup of viewfinder-open on a rented Linux GPU box (H100/A100 80 GB recommended).
#
#   git clone <this repo> /workspace/viewmaxxing
#   bash /workspace/viewmaxxing/viewfinder/setup_gpu.sh            # everything
#   bash /workspace/viewmaxxing/viewfinder/setup_gpu.sh core hyw   # only some parts
#
# Parts: base core hyw lyra. Each part is skipped if already done (delete its marker in
# $VF_WORK/.vf_done to redo it). Put $VF_WORK on the provider's persistent volume, so a
# restarted or re-rented pod keeps the ~150 GB of environments and model weights.
#
# Lyra 2.0 weights are under NVIDIA's research-only licence and HY-World under Tencent's
# community licence (not available in the EU, UK or South Korea). Read both before downloading.
set -euo pipefail

WORK="${VF_WORK:-/workspace}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARTS="${*:-base core hyw lyra}"
CONDA_DIR="$WORK/miniconda3"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export PIP_CACHE_DIR="$WORK/pip_cache"
DONE="$WORK/.vf_done"
mkdir -p "$WORK" "$DONE" "$HF_HOME"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
want() { [[ " $PARTS " == *" $1 "* ]] && [[ ! -f "$DONE/$1" ]]; }
done_part() { touch "$DONE/$1"; say "$1: done"; }

nvidia-smi --query-gpu=name,memory.total --format=csv || { echo "No NVIDIA GPU visible"; exit 1; }

# ---------------------------------------------------------------- base: system + conda
if want base; then
  say "base: system packages and Miniconda"
  if command -v apt-get >/dev/null; then
    SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq git git-lfs curl ffmpeg build-essential libgl1 libglib2.0-0 tmux >/dev/null
  fi
  if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  fi
  "$CONDA_DIR/bin/conda" config --set channel_priority flexible
  "$CONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  "$CONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
  done_part base
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# ---------------------------------------------------------------- core: server, gsplat, export
if want core; then
  say "core: vfcore env (job server, COLMAP, gsplat refine, splat-transform, collider)"
  conda env list | grep -q '^vfcore ' || conda create -y -n vfcore python=3.11 -c conda-forge nodejs=22 ffmpeg
  conda activate vfcore
  conda install -y -c nvidia/label/cuda-12.8.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev   # nvcc for gsplat's JIT build
  export CUDA_HOME="$CONDA_PREFIX" TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
  pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  pip install -q gsplat==1.5.3 viser tyro "torchmetrics[image]" tensorly splines scikit-learn matplotlib \
      tensorboard pyyaml imageio[ffmpeg] opencv-python-headless plyfile trimesh fast-simplification \
      scikit-image scipy pillow pillow-heif fastapi "uvicorn[standard]" python-multipart \
      "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" \
      "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
  # gsplat's example COLMAP reader (a pure-python "pycolmap" fork) + numpy 2 fix
  pip install -q --no-deps "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"
  SM="$(python -c 'import importlib.util as u; print(u.find_spec("pycolmap").submodule_search_locations[0])')/scene_manager.py"
  sed -i 's/np\.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/' "$SM"
  # Official pycolmap in its own folder (same import name; used only by build_colmap.py)
  rm -rf "$WORK/sfm_libs" && pip install -q --target "$WORK/sfm_libs" pycolmap==4.2.1 numpy
  [[ -d "$WORK/gsplat" ]] || git clone -q --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git "$WORK/gsplat"
  npm install -g -s @playcanvas/splat-transform@3
  say "compiling gsplat CUDA kernels (5-10 min, once)"
  python -c 'from gsplat.cuda._backend import _C; assert _C is not None; print("gsplat OK")'
  conda deactivate
  done_part core
fi

# ---------------------------------------------------------------- hyw: WorldMirror 2.0
if want hyw; then
  say "hyw: HY-World 2.0 (WorldMirror 2.0 reconstruction)"
  [[ -d "$WORK/HY-World-2.0" ]] || git clone -q https://github.com/Tencent-Hunyuan/HY-World-2.0 "$WORK/HY-World-2.0"
  conda env list | grep -q '^hyworld2 ' || conda create -y -n hyworld2 python=3.11
  conda activate hyworld2
  cd "$WORK/HY-World-2.0"
  pip install -q -r requirements.txt
  pip install -q git+https://github.com/nerfstudio-project/gsplat.git
  pip install -q flash-attn --no-build-isolation || echo "flash-attn wheel failed; WorldMirror falls back to PyTorch attention"
  say "downloading WorldMirror 2.0 weights"
  python -c "from hyworld2.worldrecon.pipeline import WorldMirrorPipeline; WorldMirrorPipeline.from_pretrained('tencent/HY-World-2.0'); print('WorldMirror OK')"
  conda deactivate
  cd "$REPO"
  done_part hyw
fi

# ---------------------------------------------------------------- lyra: Lyra 2.0 (follows Lyra-2/INSTALL.md)
if want lyra; then
  say "lyra: NVIDIA Lyra 2.0 (generative views; 30-60 min, mostly compiling flash-attn)"
  [[ -d "$WORK/lyra" ]] || git clone -q --recursive https://github.com/nv-tlabs/lyra.git "$WORK/lyra"
  cd "$WORK/lyra/Lyra-2"
  conda env list | grep -q '^lyra2 ' || conda create -y -n lyra2 python=3.10 pip cmake ninja libgl ffmpeg packaging -c conda-forge
  conda activate lyra2
  CONDA_BACKUP_CXX="" conda install -y gcc=13.3.0 gxx=13.3.0 eigen zlib -c conda-forge
  conda install -y cuda -c nvidia/label/cuda-12.8.0
  export CUDA_HOME="$CONDA_PREFIX"
  pip install -q torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
  SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
  export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  pip install -q --no-deps -r requirements.txt
  pip install -q "git+https://github.com/microsoft/MoGe.git"
  pip install -q --no-build-isolation "transformer_engine[pytorch]"
  ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"
  MAX_JOBS="${MAX_JOBS:-16}" pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
  USE_SYSTEM_EIGEN=1 pip install -q --no-build-isolation -e 'lyra_2/_src/inference/vipe'
  pip install -q --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'
  pip install -q huggingface_hub
  say "downloading Lyra 2.0 checkpoints (large)"
  huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
  PYTHONPATH=. python -c "import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1; print('Lyra imports OK, CUDA', torch.cuda.is_available())"
  conda deactivate
  cd "$REPO"
  done_part lyra
fi

cat <<EOF

All requested parts are installed ($PARTS).

Start the server (inside tmux so it survives your SSH session closing):
  tmux new -s vf
  source $CONDA_DIR/etc/profile.d/conda.sh && conda activate vfcore
  cd $REPO/viewfinder && VF_WORK=$WORK uvicorn vf.server:app --host 127.0.0.1 --port 8000

Then on YOUR computer:   ssh -L 8000:localhost:8000 <user>@<pod-address> -p <ssh-port>
and open http://localhost:8000
EOF
