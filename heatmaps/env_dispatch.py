"""
env_dispatch.py
================
Some model backends cannot live in the same Python environment as the rest
of the repo:

    * SAM-HQ2 (SysCV/sam-hq/sam-hq2) installs a top-level package literally
      named ``sam2`` -- the same import name as the official SAM2.1 package
      (facebookresearch/sam2) this repo already depends on. Two different
      packages cannot both be importable as ``sam2`` in one interpreter.
    * SAM3 (facebookresearch/sam3) requires Python >=3.12 and torch >=2.7,
      incompatible with this repo's pinned torch==1.13.1 (see
      INSTALL_FIXES.md -- pinned specifically to avoid "CUDA driver too old"
      errors with newer torch).

``scripts/setup_repo.sh`` creates one conda env per backend (see MODEL_ENV
below for the names). Rather than hand-import across incompatible envs, a
script that is asked to run with one of these backends re-execs itself
inside the correct env via ``conda run`` and exits -- the child process
does the real work with the right packages on its path.

Usage (top of a script's main(), right after parse_args()):

    from heatmaps.env_dispatch import maybe_dispatch_to_env
    maybe_dispatch_to_env(args.model_name, __file__)
"""

from __future__ import annotations

import os
import subprocess
import sys

# model_name -> conda env name created by scripts/setup_repo.sh.
# Models not listed here (SAM, SAM2.1, SAM-HQ, ...) run in the current env.
MODEL_ENV: dict[str, str] = {
    "SAM-HQ2": "sam_hq2",
    "SAM3": "sam3",
}

_DISPATCH_GUARD_ENV = "_BREPS_DISPATCHED_ENV"


def maybe_dispatch_to_env(model_name: str, script_path: str) -> None:
    """If *model_name* needs a different conda env than the current one,
    re-exec this script inside that env (via ``conda run``) and never
    return -- the parent process exits with the child's return code.

    No-op when the model runs fine in the current env, or when we are
    already the re-exec'd child (guarded by an env var so this can't loop).
    """
    env_name = MODEL_ENV.get(model_name)
    if env_name is None:
        return

    if os.environ.get(_DISPATCH_GUARD_ENV) == env_name:
        # Already running inside the target env (re-exec'd child).
        return

    print(
        f"[env_dispatch] --model_name {model_name} requires conda env "
        f"'{env_name}' (created by scripts/setup_repo.sh); re-launching "
        f"there ...",
        file=sys.stderr,
    )

    child_env = os.environ.copy()
    child_env[_DISPATCH_GUARD_ENV] = env_name

    cmd = ["conda", "run", "-n", env_name, "--no-capture-output",
           "python", script_path, *sys.argv[1:]]
    try:
        result = subprocess.run(cmd, env=child_env)
    except FileNotFoundError as e:
        raise SystemExit(
            f"[env_dispatch] could not find 'conda' on PATH to launch env "
            f"'{env_name}': {e}\n"
            f"Run scripts/setup_repo.sh first, or activate '{env_name}' "
            f"manually and re-run without going through env_dispatch."
        )
    raise SystemExit(result.returncode)
