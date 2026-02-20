#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${ENV_PATH:-${HOME}/conda_envs/nanogpt_env}"

if [[ -n "${CONDA_SH:-}" ]]; then
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "CONDA_SH is set, but file was not found: ${CONDA_SH}" >&2
    exit 1
  fi
else
  for candidate in \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      CONDA_SH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CONDA_SH:-}" || ! -f "${CONDA_SH}" ]]; then
  cat >&2 <<'EOF'
Could not locate conda initialization script.
Set CONDA_SH to your conda.sh path and retry, for example:
  export CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
  source activate_env.sh
EOF
  exit 1
fi

source "${CONDA_SH}"
conda activate "${ENV_PATH}"
