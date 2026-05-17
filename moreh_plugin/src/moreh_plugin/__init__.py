"""Demo OOT plugin for SGLang, scaffolded for AMD ROCm.

Nothing in this package runs unless SGLang's plugin loader discovers it via
the entry points declared in pyproject.toml. See README.md for the map of
which file demonstrates which extension mechanism.
"""

import logging

logger = logging.getLogger("moreh_plugin")
