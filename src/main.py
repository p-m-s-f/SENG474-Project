import argparse
import subprocess
import sys
from pathlib import Path


def main():
    scripts = {
        "prepare": "prepare_data.py",
        "train": "train.py",
        "predict": "predict.py",
        "demo": "demo.py",
        "evaluate": "evaluate.py",
    }

    parser = argparse.ArgumentParser(
        description="Wildfire classification project",
        epilog=(
            "Example: python src/main.py train --stage risk\n"
            "Use COMMAND --help for that command's options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=scripts)
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Options passed to the selected script",
    )
    args = parser.parse_args()

    script = Path(__file__).resolve().parent / scripts[args.command]

    result = subprocess.run([
        sys.executable,
        str(script),
        *args.arguments,
    ])

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())