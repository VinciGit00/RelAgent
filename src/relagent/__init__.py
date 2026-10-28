"""Relational Benchmarking Agent powered by CAMEL-AI."""

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT.parent.parent / ".env")

