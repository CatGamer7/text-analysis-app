"""Build script to automate packaging the backend using PyInstaller."""
import subprocess
import sys
from pathlib import Path


def main():
    root_dir = Path(__file__).parent.parent.resolve()
    spec_file = root_dir / "packaging" / "backend.spec"

    print("=" * 60)
    print("Building Text Analysis Backend Executable Distribution")
    print(f"Project root: {root_dir}")
    print(f"Spec file:    {spec_file}")
    print("=" * 60)

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[!] PyInstaller is not installed in the current environment.")
        print("    Install it via: pip install pyinstaller")
        print("    Then re-run this script: python packaging/build_backend.py")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(root_dir))
    if result.returncode == 0:
        print("\n[+] Build successful! Output located at:")
        print(f"    {root_dir / 'dist' / 'text-analysis-backend'}")
    else:
        print(f"\n[-] Build failed with exit code: {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
