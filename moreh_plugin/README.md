# moreh_plugin (demo)

A **boilerplate out-of-tree (OOT) plugin** for SGLang, scaffolded around AMD
ROCm as the target platform. 

Use it as a starting point for a real OOT integration (custom hardware,
custom kernels, custom serving logic, custom templates, etc.).

## Install (editable, against the in-tree sglang checkout)

```bash
pip install -e ./moreh_plugin
```

After install, `pip show moreh_plugin` should list the entry points,
and any SGLang process will pick them up automatically at startup.

## Try it

```bash
# Force selection of this platform (if multiple are installed):
export SGLANG_PLATFORM=moreh_platform

# Optional: restrict which *general* plugins load:
export SGLANG_PLUGINS=moreh_hooks,moreh_backends,moreh_templates

# Now run any sglang entrypoint — the logger lines below will show up:
python -m sglang.launch_server --model-path <some-model> --attention-backend moreh
```

Look for log lines starting with `[moreh-plugin]` to confirm each hook fired.
