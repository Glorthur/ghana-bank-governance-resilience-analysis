import subprocess
from pathlib import Path

from conftest import ROOT


def tracked_files():
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [Path(line) for line in output.splitlines() if line]


def test_no_forbidden_source_or_binary_artifacts():
    forbidden_suffixes = {".pdf", ".docx", ".xlsx", ".xlsm", ".zip", ".pyc"}
    forbidden_names = {"__pycache__"}
    files = tracked_files()
    assert not [p for p in files if p.suffix.lower() in forbidden_suffixes or any(part in forbidden_names for part in p.parts)]


def test_no_machine_local_paths_in_tracked_text():
    path_markers = ("G:\\\\", "C:\\\\", "/Users/", "/home/", "\\\\Users\\\\")
    for relative in tracked_files():
        if relative.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".ico"}:
            continue
        content = (ROOT / relative).read_bytes()
        if b"\x00" in content:
            continue
        text = content.decode("utf-8", errors="ignore")
        assert not any(marker in text for marker in path_markers), relative


def test_scripts_do_not_reference_drive_archive():
    for script in (ROOT / "scripts").glob("*.py"):
        text = script.read_text(encoding="utf-8")
        assert "G:" not in text
        assert "My Drive" not in text

