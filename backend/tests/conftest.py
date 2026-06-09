"""Pytest configuration — ensures the backend root is on sys.path."""
import sys
import os

# Add the backend directory so all imports work without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
