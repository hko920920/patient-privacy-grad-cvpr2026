#!/usr/bin/env python3
"""Verify the Compiler V3.9 reviewer-facing V9 code/data package."""

from __future__ import annotations

try:
    from scripts import verify_public_supplement_v39 as core
except (ImportError, ModuleNotFoundError):
    import verify_public_supplement_v39 as core


core.PACKAGE_ID = "compiler_code_and_data_supplement_aaai27_v9"


if __name__ == "__main__":
    core.main()
