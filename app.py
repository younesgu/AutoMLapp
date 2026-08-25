"""Streamlit entry point for AutoMLapp."""

import sys
from pathlib import Path


def run() -> None:
    """Add the src directory to the import path and start Streamlit."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from automl_app.app import main

    main()


if __name__ == "__main__":
    run()
