#!/usr/bin/env python3
"""
Installation script for Bangla LLM package with all dependencies.
This script handles the special installation requirements for GPU support and Git packages.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"{'=' * 50}")
    print(f"Running: {description or cmd}")
    print(f"{'=' * 50}")

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main installation function."""

    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ):
        print("Warning: Not in a virtual environment. Consider using venv or conda.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    print("Starting Bangla LLM installation...")

    # Step 1: Install core package
    success = run_command(
        "pip install -e .",
        "Installing core Bangla LLM package"
    )
    if not success:
        print("Failed to install core package")
        sys.exit(1)

    # Step 2: Install GPU support for llama-cpp-python
    gpu_support = input("Install GPU support for llama-cpp-python? (y/N): ")
    if gpu_support.lower() == 'y':
        success = run_command(
            "pip install llama-cpp-python==0.2.85 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122",
            "Installing llama-cpp-python with CUDA support"
        )
        if not success:
            print("Warning: Failed to install GPU support. Continuing with CPU version.")
            run_command("pip install llama-cpp-python==0.2.85", "Installing CPU version")

    # Step 3: Install Bangla normalizer from git
    success = run_command(
        "pip install git+https://github.com/csebuetnlp/normalizer.git",
        "Installing Bangla text normalizer from GitHub"
    )
    if not success:
        print("Warning: Failed to install Bangla normalizer")

    # Step 4: Download NLTK data
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Warning: Failed to download NLTK data: {e}")

    # Step 5: Verify installation
    print("\n" + "=" * 50)
    print("Verifying installation...")
    print("=" * 50)

    try:
        # Test imports
        import pandas
        import numpy
        import torch
        import transformers
        import langchain
        print("✓ Core dependencies installed successfully")

        # Check if GPU support is available
        if torch.cuda.is_available():
            print(f"✓ CUDA available with {torch.cuda.device_count()} GPU(s)")
        else:
            print("ℹ CUDA not available (CPU-only mode)")

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    print("\n" + "=" * 50)
    print("Installation completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Import the package: from bangla_llm import *")
    print("2. Check the documentation for usage examples")
    print("3. Run tests if available: pytest tests/")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
