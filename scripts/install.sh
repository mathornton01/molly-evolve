#!/bin/bash
# Quick install script for Molly Evolution.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/mathornton01/molly-evolve/main/scripts/install.sh | bash
#   # or
#   bash scripts/install.sh

set -e

echo "========================================"
echo "  Molly Evolution — Quick Install"
echo "========================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.9+ first."
    exit 1
fi

PYTHON=$(command -v python3)
echo "Python: $($PYTHON --version)"

# Check pip
if ! $PYTHON -m pip --version &> /dev/null; then
    echo "ERROR: pip not found. Install pip first."
    exit 1
fi

# Clone or update repo
if [ -d "molly-evolve" ]; then
    echo "Updating existing installation..."
    cd molly-evolve
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/mathornton01/molly-evolve.git
    cd molly-evolve
fi

# Install core package
echo
echo "Installing molly-evolution..."
$PYTHON -m pip install -e . --quiet

# Install optional deps for comparison methods
echo "Installing comparison method dependencies..."
$PYTHON -m pip install peft bitsandbytes datasets accelerate sentencepiece --quiet

# Verify installation
echo
echo "Verifying..."
$PYTHON -c "from molly_evolution import DualGenome, GeneScorer; print('  Core: OK')"
$PYTHON -c "from molly_evolution.methods import list_methods; print(f'  Methods: {list_methods()}')"
$PYTHON -c "from molly_evolution.cli import main; print('  CLI: OK')"

# Check GPU
$PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print('  GPU: not available (CPU mode)')
"

echo
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo
echo "Quick start:"
echo "  molly evolve --model gpt2 --domains code,legal --quicktest"
echo "  molly compare --model gpt2 --methods gene-conv,lora --quicktest"
echo "  molly info 7b"
echo "  molly benchmark"
echo
