# Notes for AI agents

Human-facing dev environment docs live in [DEVELOPMENT.md](./DEVELOPMENT.md).
**Do not duplicate that content here.** This file is for agent-only context.

## Fork context

- This is a fork of [sgl-project/sglang](https://github.com/sgl-project/sglang).
- Add a plugin at `moreh_plugin/` (treat as first-class in-tree code, not a
  stray directory) and is configured for AMD ROCm only. New code should be
  written in the plugin.
- The `python/pyproject.toml` swap (ROCm variant over upstream's CUDA variant)
  is intentional and committed on this branch. Don't propose reverting it or
  flag the deleted `python/pyproject_other.toml` as suspicious.

## Where commands run

Most build / install / test / launch-server commands only work **inside** the
ROCm dev container (`docker/compose.rocm.yaml`), not on the host. When
proposing such a command:

1. Assume CWD = `/sgl-workspace/sglang` inside the container unless the user
   is clearly doing host-side pre-flight (e.g. `docker compose build`).
2. If the conversation context is ambiguous about whether the user is inside
   or outside the container, ask before suggesting a long-running build.

The host typically lacks the ROCm/HIP toolchain; don't suggest `pip install
sgl-kernel`-style commands directly on the host.

## Team decisions encoded in the setup

- **`container_name` uses `$USER`; `image` tag uses `$GPU_ARCH`.** Don't
  hardcode either when editing the compose file.
- **`docker/compose.yaml` (upstream CUDA example) stays untouched** for
  upstream parity. Add to `compose.rocm.yaml` instead.

## File ownership

The container runs as root. Files written from inside the container land on
the host owned by `root:root`. If the user hits permission errors editing
build artifacts (`build/`, `__pycache__`, `*.so`), the fix is
`sudo chown -R $USER:$USER .`, not changing the compose `user:` directive
(that would break the shared-image model).

## Reference material

- Upstream platform / feature docs: `docs/` (especially
  `docs/platforms/amd_gpu.md` and `docs/developer_guide/`).
- Most generic "how does sglang feature X work" questions are answered
  upstream — search `docs/` before assuming fork-specific behavior.
