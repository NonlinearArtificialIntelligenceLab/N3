import os
import yaml
import argparse
import subprocess
from itertools import product

def load_sweep_config(sweep_name):
    with open("config/sweeps.yaml") as f:
        configs = yaml.safe_load(f)
    return configs[sweep_name]

def generate_commands(config):
    """Generate all parameter combinations"""
    keys = list(config['parameters'].keys())
    values = [config['parameters'][k]['values'] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]

def get_script_path(sweep_name):
    match sweep_name:
        case "bessel_regression":
            return "benchmarking/bessel_standard.py"
        case "spiral_classification":
            return "benchmarking/spiral_standard.py"
        case "bessel_regression_baseline":
            return "benchmarking/bessel_static.py"
        case "spiral_classification_baseline":
            return "benchmarking/spiral_static.py"
        case "gaussian_regression":
            return "benchmarking/gaussian_standard.py"
        case "gaussian_regression_baseline":
            return "benchmarking/gaussian_static.py"
        case _:
            raise ValueError(f"Unknown sweep name: {sweep_name}")

def run_sweep(sweep_name, max_parallel=4):
    config = load_sweep_config(sweep_name)
    commands = generate_commands(config)
    script_path = get_script_path(sweep_name)
    processes = []
    for i, params in enumerate(commands):
        # Build command string
        cmd = [
            "python",
            script_path,
            "--wandb",
            f"--group={sweep_name}",
            f"--exp_name={sweep_name}",
            f"--epochs={config['epochs']}",
            f"--num_runs={config['num_runs']}",
            f"--base_seed={config['base_seed']}",
        ]
        for k, v in params.items():
            cmd.append(f"--{k}={v}")

        # Assign GPU in round-robin fashion
        gpu_id = i % max_parallel
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Start process
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

        # Throttle parallel processes
        if (i+1) % max_parallel == 0:
            [p.wait() for p in processes]
            processes = []

    # Wait for remaining processes
    [p.wait() for p in processes]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_name", choices=["bessel_regression", "spiral_classification", "bessel_regression_baseline", "spiral_classification_baseline", "gaussian_regression", "gaussian_regression_baseline"])
    parser.add_argument("--max_parallel", type=int, default=4)
    args = parser.parse_args()

    run_sweep(args.sweep_name, args.max_parallel)
