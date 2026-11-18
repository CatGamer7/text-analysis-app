"""One-click master build script for both Backend and Desktop Frontend."""
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
FRONTEND_DIR = ROOT_DIR / "frontend"


def main():
    print("=" * 60)
    print(" [1/2] Building Backend Distribution (PyInstaller)")
    print("=" * 60)
    spec_path = ROOT_DIR / "packaging" / "backend.spec"
    py_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_path)
    ]
    res_be = subprocess.run(py_cmd, cwd=str(ROOT_DIR))
    if res_be.returncode != 0:
        print("[!] Backend build failed!")
        sys.exit(res_be.returncode)

    print("\n" + "=" * 60)
    print(" [2/2] Packaging Desktop Application (Electron)")
    print("=" * 60)
    res_fe = subprocess.run(["npx.cmd", "electron-builder", "--dir"], cwd=str(FRONTEND_DIR), shell=True)
    if res_fe.returncode != 0:
        print("[!] Frontend package build failed!")
        sys.exit(res_fe.returncode)

    print("\n" + "=" * 60)
    print(" [OK] Build Complete!")
    print(f" Executable: {FRONTEND_DIR / 'release' / 'win-unpacked' / 'TextAnalysisApp.exe'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
