"""
Runs the web front-end's JavaScript unit tests with Node's built-in runner.

``node --test medbridge/web/tests`` exercises the escape-first Markdown
renderer (XSS attempts, javascript: links, nested Markdown), the POST-SSE
parser, the localStorage wrapper and the payload normalisers. Skipped when
Node.js is not on PATH (it is not a runtime dependency of MedBridge).
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from medbridge.web import WEB_DIR

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed; JS unit tests skipped")
def test_node_unit_tests_pass() -> None:
    tests_dir = WEB_DIR / "tests"
    result = subprocess.run(
        [str(NODE), "--test", str(tests_dir)],
        cwd=WEB_DIR.parent.parent,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "# fail 0" in output or "ℹ fail 0" in output, output[-4000:]
