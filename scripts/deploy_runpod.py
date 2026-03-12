"""
Deploy Molly Evolution to RunPod.

Creates a RunPod pod with the right GPU, installs dependencies,
and runs the specified experiment.

Prerequisites:
  pip install runpod
  export RUNPOD_API_KEY=your_key_here

Usage:
  python scripts/deploy_runpod.py --model gpt2 --gpu a100-80gb
  python scripts/deploy_runpod.py --model meta-llama/Llama-2-7b-hf --gpu a100-80gb
  python scripts/deploy_runpod.py --model gpt2 --command "molly compare --quicktest"
"""

import argparse
import os
import sys
import json

# GPU templates for RunPod
GPU_CONFIGS = {
    "a100-80gb": {
        "gpu_type": "NVIDIA A100 80GB PCIe",
        "gpu_count": 1,
        "volume_size": 50,
        "container_disk": 20,
        "min_ram": 64,
    },
    "a100-40gb": {
        "gpu_type": "NVIDIA A100-SXM4-40GB",
        "gpu_count": 1,
        "volume_size": 50,
        "container_disk": 20,
        "min_ram": 64,
    },
    "a6000": {
        "gpu_type": "NVIDIA RTX A6000",
        "gpu_count": 1,
        "volume_size": 30,
        "container_disk": 20,
        "min_ram": 32,
    },
    "4090": {
        "gpu_type": "NVIDIA GeForce RTX 4090",
        "gpu_count": 1,
        "volume_size": 30,
        "container_disk": 20,
        "min_ram": 32,
    },
    "2xa100": {
        "gpu_type": "NVIDIA A100 80GB PCIe",
        "gpu_count": 2,
        "volume_size": 100,
        "container_disk": 30,
        "min_ram": 128,
    },
    "4xa100": {
        "gpu_type": "NVIDIA A100 80GB PCIe",
        "gpu_count": 4,
        "volume_size": 200,
        "container_disk": 50,
        "min_ram": 256,
    },
}

SETUP_SCRIPT = """#!/bin/bash
set -e

echo "=== Molly Evolution Setup ==="

# Install from GitHub
pip install git+https://github.com/mathornton01/molly-evolve.git
pip install peft bitsandbytes datasets accelerate sentencepiece

# Verify
python -c "from molly_evolution import DualGenome; print('Molly Evolution installed')"

echo "=== Setup complete ==="
"""


def generate_launch_script(model, command, domains="code,legal,medical"):
    """Generate the experiment launch script."""
    if command:
        return f"""#!/bin/bash
set -e
{SETUP_SCRIPT}
echo "=== Running experiment ==="
{command}
echo "=== Done ==="
"""

    return f"""#!/bin/bash
set -e
{SETUP_SCRIPT}

echo "=== Running Molly Evolution ==="

# Run benchmark first
python -m molly_evolution.cli benchmark

# Run comparison
python -m molly_evolution.cli compare \\
    --model {model} \\
    --methods gene-conv,lora,qlora \\
    --domains {domains} \\
    --epochs 3

echo "=== Experiment complete ==="
"""


def deploy_runpod(args):
    """Deploy to RunPod using their API."""
    try:
        import runpod
    except ImportError:
        print("RunPod SDK not installed. Install with: pip install runpod")
        print("\nAlternatively, use the generated script manually:")
        print_manual_instructions(args)
        return

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY not set.")
        print("Set it with: export RUNPOD_API_KEY=your_key_here")
        print("\nAlternatively, use the generated script manually:")
        print_manual_instructions(args)
        return

    runpod.api_key = api_key
    gpu_config = GPU_CONFIGS.get(args.gpu, GPU_CONFIGS["a100-80gb"])

    print(f"Deploying to RunPod...")
    print(f"  GPU: {gpu_config['gpu_type']} x{gpu_config['gpu_count']}")
    print(f"  Model: {args.model}")
    print()

    launch_script = generate_launch_script(args.model, args.command)

    try:
        pod = runpod.create_pod(
            name=f"molly-evolution-{args.model.split('/')[-1]}",
            image_name="nvidia/cuda:12.1.1-devel-ubuntu22.04",
            gpu_type_id=gpu_config["gpu_type"],
            gpu_count=gpu_config["gpu_count"],
            volume_in_gb=gpu_config["volume_size"],
            container_disk_in_gb=gpu_config["container_disk"],
            docker_args=f"bash -c '{launch_script}'",
        )
        print(f"Pod created: {pod['id']}")
        print(f"Status: {pod.get('desiredStatus', 'starting')}")
    except Exception as e:
        print(f"Error creating pod: {e}")
        print("\nManual instructions:")
        print_manual_instructions(args)


def print_manual_instructions(args):
    """Print manual RunPod setup instructions."""
    gpu_config = GPU_CONFIGS.get(args.gpu, GPU_CONFIGS["a100-80gb"])

    print(f"""
Manual RunPod Setup:

1. Go to https://runpod.io/console/pods
2. Create a new pod:
   - GPU: {gpu_config['gpu_type']}
   - Count: {gpu_config['gpu_count']}
   - Image: nvidia/cuda:12.1.1-devel-ubuntu22.04
   - Disk: {gpu_config['container_disk']} GB
   - Volume: {gpu_config['volume_size']} GB

3. SSH into the pod and run:
   pip install git+https://github.com/mathornton01/molly-evolve.git
   pip install peft bitsandbytes datasets accelerate sentencepiece

4. Run experiment:
   python -m molly_evolution.cli compare \\
       --model {args.model} \\
       --methods gene-conv,lora,qlora \\
       --domains code,legal,medical
""")


def main():
    parser = argparse.ArgumentParser(description="Deploy Molly Evolution to RunPod")
    parser.add_argument("--model", default="gpt2",
                        help="Model name (default: gpt2)")
    parser.add_argument("--gpu", default="a100-80gb",
                        choices=list(GPU_CONFIGS.keys()),
                        help="GPU type")
    parser.add_argument("--command", default=None,
                        help="Custom command to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print setup script without deploying")

    args = parser.parse_args()

    if args.dry_run:
        print(generate_launch_script(args.model, args.command))
        return

    deploy_runpod(args)


if __name__ == "__main__":
    main()
