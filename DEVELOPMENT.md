# Dev environment (ROCm)

This fork ships a `moreh_plugin/` add-on. The canonical workflow is a
long-lived ROCm container defined by `docker/compose.rocm.yaml`. The container
is per-user (`sglang-rocm-$USER`); the image is shared per GPU/ROCm flavor.

## Quick start

```bash
cd docker/
docker compose -f compose.rocm.yaml build sglang-rocm    # first time only, ~30+ min
docker compose -f compose.rocm.yaml up -d
docker compose -f compose.rocm.yaml exec sglang-rocm bash
```

All compose subcommands assume CWD = `docker/`. The bind mount puts your host
checkout at `/sgl-workspace/sglang` inside the container, so host edits are
live for Python and visible to in-container builds for C++/Rust.

## What the image gives you vs. what you install editable

The Dockerfile (`docker/rocm.Dockerfile`) installs every component at build
time, but the bind mount overlays the cloned tree, so anything installed
non-editably (HIP `.so`, Rust binaries, Python wheels) is **shadowed** by your
host source. Treat the in-image install as a fallback for "image without bind
mount" only.

For a working editable dev env, inside the container:

| Component | One-time setup | Rebuild trigger |
|-----------|----------------|-----------------|
| `python/sglang` | already editable from Dockerfile | none — pure-Python edits are live |
| `sgl-kernel` | `pip uninstall -y sgl-kernel sgl_kernel && cd sgl-kernel && cp pyproject_rocm.toml pyproject.toml && AMDGPU_TARGET=gfx942 python setup_rocm.py develop` | `python setup_rocm.py build_ext --inplace` after `.hip`/`.cu`/`.cpp` changes |
| `moreh_plugin` | `pip install -e moreh_plugin/` | none — pure-Python |
| `rust/sglang-grpc` | bundled with `python/sglang`'s editable install | `cd python && pip install --no-deps -e .` after `.rs` / `proto/` changes |
| `sgl-model-gateway` (bindings) | `cd sgl-model-gateway/bindings/python && ulimit -n 65536 && maturin develop --release --features vendored-openssl` | re-run `maturin develop --release` after `.rs` changes |
| `sgl-model-gateway` (binary) | image's prebuilt binary at `/usr/local/bin/sgl-model-gateway` is usually fine | `cd sgl-model-gateway && cargo build --release --bin sgl-model-gateway` |

Verify the editable install with `python -c "import sgl_kernel; print(sgl_kernel.__file__)"`;
the path must live under `/sgl-workspace/sglang/...`, not `/usr/local/lib/...`.

## Pre-built bits worth knowing

- `pyproject.toml` swap: the rocm Dockerfile copies `pyproject_rocm.toml` over
  `pyproject.toml` for both `python/` and `sgl-kernel/`. For `python/`, the
  swap is already committed on this branch (see git status). For `sgl-kernel/`
  you have to do the swap yourself in the editable setup above.
- `/etc/gitconfig` is injected via compose `configs:` to mark the mount as a
  safe directory — works regardless of git version inside the image.
- Shared data path `/remote/vast0/share-mv` is bind-mounted as-is; the compose
  file assumes that path exists on the host. If you're not on a moreh dev box
  with that mount, comment the volume out.

## Common gotchas

1. **Port collisions under `network_mode: host`.** Two devs on the same host
   both launching `sglang.launch_server --port 30000` will collide. Coordinate
   ports informally (e.g. `30000 + UID % 100`).
2. **Root-owned files in the working tree.** The container runs as root, so
   build artifacts (`build/`, `__pycache__`, `compile_commands.json`, `.so`s)
   land on the host owned by `root:root`. Clean with
   `sudo chown -R $USER:$USER .` periodically.
3. **Stale editable install after `docker compose down`.** Editable installs
   are baked into the **image's** site-packages, not the bind mount. A
   `down` + `up` reuses the same container and editable state. A
   `down -v` or recreate wipes them, and you re-run the editable-install
   commands above.
4. **`SGL_BRANCH` build arg is dead under the bind mount** — the cloned repo is
   overlaid by your local checkout. Only matters if you're using the image
   without a bind mount.
