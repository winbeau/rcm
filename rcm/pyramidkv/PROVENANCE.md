# pyramidkv — vendored

Verbatim copy of `pyramidkv/` from **IF-LAB-PKU/Pyramid-Forcing**, commit
`3f403f7` (`pyramidkv/` last touched by `4c0ebda`, "refactor: rename headkv to
pyramidkv across the codebase"). Vendored 2026-07-26.

## Why vendored rather than a path or submodule dependency

The dependency closure is `torch`, `numpy`, `csv`, and an optional `triton`
guarded by try/except — nothing from Self-Forcing's `wan` package — so a copy is
self-contained. A path dependency would drag in Pyramid-Forcing's much heavier
requirements and tie the two checkouts to the same machine layout; a submodule
would put a 4 GB repo three levels deep. The port also wants a pinned version:
this is the code `configs/head_configs/best_labels.csv` and the paper's numbers
correspond to, and it should not drift under us mid-comparison.

## The one permitted edit

`__init__.py` registers `sys.modules["pyramidkv"]` as an alias for this package.
Three sites in the opt-in C++ shadow path (`_cpp_shadow.py:63`,
`adaptive_cache.py:417`, `adaptive_cache.py:460`) import the package by its
absolute name, which stops resolving once it lives at `rcm.pyramidkv`. The alias
fixes all three without touching them, so re-vendoring stays a plain rsync and
every other file remains byte-identical to upstream.

## Rules

**Do not edit these files** beyond the shim above. Any rCM-specific behaviour belongs in the port glue
(`rcm/utils/pyramid_attention.py`, `experiments/pyramid_port/`), not here. If a
change is genuinely needed upstream, make it in Pyramid-Forcing and re-vendor —
otherwise an rCM-vs-Self-Forcing comparison stops being a backbone comparison
and becomes an implementation comparison.

To re-vendor:

```bash
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
    <path-to>/Pyramid-Forcing/pyramidkv/ rcm/pyramidkv/
```

then update the commit hash above.

## What is opt-in here

Everything under `csrc/`, `_cpp_*`, `_scatter_ext`, and the MegaCache path is
**off by default** and needs an environment variable to activate
(`PYRAMIDKV_USE_CPP_PACK`, `PYRAMIDKV_FORCE_SCATTER`, `PYRAMIDKV_USE_MEGA_CACHE`).
The default path is pure PyTorch plus `flash_attn_varlen_func`, which is why the
port needs no kernel compilation. `pyramidkv/rope.py` will use Triton if it
imports and fall back to PyTorch otherwise.
