"""PyramidKV frame selection strategies.

This package implements the [sink ... middle ... recent] architecture
for per-head KV cache management. Three middle strategies are available
and can be combined (union):

- CyclicStrategy: t mod T phase-bucket anchors
- LagStrategy: fixed-offset t-k anchors
- StrideStrategy: every k-th frame anchors
- MergeStrategy: spatiotemporal patch-block merging

HeadComposition ties sink_frames + middle strategies + recent_frames together
for each attention head. The factory module builds compositions from YAML config.
"""

# --- vendoring shim (the ONLY edit to this package; see PROVENANCE.md) -------
# Three sites in the C++ shadow path import the package by its absolute name
# (`from pyramidkv._cpp_shadow import ...`), which does not resolve once the
# package lives at `rcm.pyramidkv`. Aliasing here keeps every other file
# byte-identical to upstream, so re-vendoring stays a plain rsync.
import sys as _sys

_sys.modules.setdefault("pyramidkv", _sys.modules[__name__])
# -----------------------------------------------------------------------------

from .base import FrameAnchor, HeadComposition, MiddleStrategy
from .cyclic import CyclicStrategy
from .lag import LagStrategy
from .stride import StrideStrategy
from .merge import MergeStrategy
from .recent import RecentStrategy
from .factory import (
    HEAD_LABEL_MAP,
    build_compositions,
    load_head_labels,
)
from .config import PyramidKVConfig
from .cache import PyramidKVCache
from .adaptive_cache import AdaptiveKVCache

__all__ = [
    "FrameAnchor",
    "HeadComposition",
    "MiddleStrategy",
    "CyclicStrategy",
    "LagStrategy",
    "StrideStrategy",
    "MergeStrategy",
    "RecentStrategy",
    "HEAD_LABEL_MAP",
    "build_compositions",
    "load_head_labels",
    "PyramidKVConfig",
    "PyramidKVCache",
    "AdaptiveKVCache",
]
