"""Use mjlab's real terrain generator on a laptop.

mjlab's package __init__ imports warp / tyro, which the local tool environment does not have. The terrain generator
itself only needs mujoco and numpy, so this module registers the terrain sub-modules of the reference mjlab source
(upstream/references/mjlab-8bb2bc9, the revision the variant locks) without running the package __init__.
Call install() before importing anything from mjlab.terrains.*.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

MJLAB_SRC = Path(__file__).resolve().parents[2] / "upstream" / "references" / "mjlab-8bb2bc9" / "src" / "mjlab"


def install() -> None:
    if "mjlab.terrains.terrain_generator" in sys.modules:
        return
    if "mjlab" not in sys.modules:
        try:
            importlib.import_module("mjlab")  # the offline stub, when it is on the path
        except Exception:
            m = types.ModuleType("mjlab")
            m.__path__ = []
            sys.modules["mjlab"] = m
    pkg = sys.modules["mjlab"]
    if str(MJLAB_SRC) not in list(pkg.__path__):
        pkg.__path__ = list(pkg.__path__) + [str(MJLAB_SRC)]
    for name, sub in (("mjlab.terrains", "terrains"), ("mjlab.utils", "utils")):
        if name in sys.modules:
            mod = sys.modules[name]
            if str(MJLAB_SRC / sub) not in list(mod.__path__):
                mod.__path__ = list(mod.__path__) + [str(MJLAB_SRC / sub)]
        else:
            try:
                mod = importlib.import_module(name) if name == "mjlab.utils" else None
            except Exception:
                mod = None
            if mod is None:
                mod = types.ModuleType(name)
                mod.__path__ = [str(MJLAB_SRC / sub)]
                sys.modules[name] = mod
            elif str(MJLAB_SRC / sub) not in list(mod.__path__):
                mod.__path__ = list(mod.__path__) + [str(MJLAB_SRC / sub)]
    for sub in ("terrain_generator", "primitive_terrains", "heightfield_terrains"):
        importlib.import_module(f"mjlab.terrains.{sub}")


def build_terrain_model(gen_cfg):
    """Compile a standalone MuJoCo model that contains only the generated terrain. Returns (model, terrain_origins)."""
    import mujoco

    from mjlab.terrains.terrain_generator import TerrainGenerator

    spec = mujoco.MjSpec()
    gen = TerrainGenerator(gen_cfg, device="cpu")
    gen.compile(spec)
    return spec.compile(), gen.terrain_origins.copy()
