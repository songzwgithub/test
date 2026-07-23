from __future__ import annotations

import subprocess
import sys


def test_cli_help() -> None:
    result = subprocess.run([sys.executable, "-m", "hengshui_insar.cli", "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "hengshui-insar" in result.stdout
