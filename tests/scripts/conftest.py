"""Load the stand-alone scripts in ``scripts/`` as modules for hermetic tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name: str) -> ModuleType:
    module_name = f"_medbridge_script_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def smoke() -> ModuleType:
    return load_script("smoke_test")


@pytest.fixture(scope="session")
def load() -> ModuleType:
    return load_script("load_test")


@pytest.fixture(scope="session")
def probe() -> ModuleType:
    return load_script("latency_probe")
