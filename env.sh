# Source from WSL (AEON-Training): sets the venv and the run environment.
source ~/.venv/bin/activate
cd "/path/to/repo"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_DATASETS_CACHE="$HOME/lm-data/hf-cache"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TF_CPP_MIN_LOG_LEVEL=2
