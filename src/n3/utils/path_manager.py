import os
from datetime import datetime
from typing import Dict, Any

def build_output_path(args: Dict[str, Any], run_idx: int = 0) -> str:
    """Construct structured output path based on parameters"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_type = "regression" if "bessel" in args['script'] else "classification"

    # Core parameters to include in path
    path_params = {
        'hidden_size': args.get('N_max', args.get('hidden_size', 10)),
        'dataset_size': args.get('n_samples', 2**15),
        'lr': args.get('learning_rate', 1e-3),
        'task': task_type,
        'exp_name': args.get('exp_name', 'default'),
        'timestamp': timestamp,
        'run_idx': run_idx,
        'seed': args['base_seed'] + run_idx
    }

    # Create path components
    path_components = [f"run_{run_idx}",
        f"seed_{path_params['seed']}"] + [
        f"{k}={v}" for k, v in path_params.items()
        if k not in ['timestamp', 'exp_name']
    ]

    # Final path structure
    path = os.path.join(
        args['out_root'],
        args['exp_name'],
        *path_components,
        timestamp
    )

    # Create directory if needed
    os.makedirs(path, exist_ok=True)
    return path
