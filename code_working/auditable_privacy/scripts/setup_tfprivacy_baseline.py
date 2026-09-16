"""Create the pinned TensorFlow Privacy comparison environment portably."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT = Path(".venv-tfprivacy-baseline")


def environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def run(interpreter: Path, *arguments: str) -> None:
    completed = subprocess.run(
        [str(interpreter), *arguments],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(
            f"Command failed with exit {completed.returncode}: "
            + " ".join(arguments)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-path",
        type=Path,
        default=DEFAULT_ENVIRONMENT,
        help="Virtual-environment directory (default: .venv-tfprivacy-baseline).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = args.environment_path
    if not environment.is_absolute():
        environment = PROJECT_ROOT / environment
    interpreter = environment_python(environment)

    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    elif not interpreter.is_file():
        raise SystemExit(
            "Environment path already exists but has no expected Python "
            f"interpreter: {interpreter}"
        )

    run(
        interpreter,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade",
        "pip==25.1.1",
    )
    run(
        interpreter,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-r",
        str(PROJECT_ROOT / "requirements-tfprivacy-baseline.txt"),
    )
    run(
        interpreter,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "tensorflow-privacy==0.9.0",
    )
    validation = (
        "import hashlib, importlib.metadata; "
        "from pathlib import Path; "
        "dist=importlib.metadata.distribution('tensorflow-privacy'); "
        "site=Path(dist.locate_file('')); "
        "module=site/'tensorflow_privacy'/'privacy'/'analysis'/"
        "'compute_dp_sgd_privacy_lib.py'; "
        "assert module.is_file(), f'Official module not found: {module}'; "
        "print('tensorflow-privacy=' + dist.version); "
        "print('dp-accounting=' + "
        "importlib.metadata.version('dp-accounting')); "
        "print('module_sha256=' + "
        "hashlib.sha256(module.read_bytes()).hexdigest())"
    )
    run(interpreter, "-c", validation)
    print(f"TensorFlow Privacy environment ready: {interpreter}")


if __name__ == "__main__":
    main()

