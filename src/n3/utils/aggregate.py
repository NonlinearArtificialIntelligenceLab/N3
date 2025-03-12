import jax.numpy as jnp
import json
from pathlib import Path

def aggregate_runs(experiment_dir: str):
    """Combine results from multiple runs"""
    metrics = []
    for run_dir in Path(experiment_dir).glob("run_*"):
        with open(run_dir / "metrics.json") as f:
            metrics.append(json.load(f))

    # Stack arrays and compute statistics
    return {
        'train_loss_mean': jnp.mean(jnp.array([m['train_losses'] for m in metrics]), axis=0),
        'train_loss_std': jnp.std(jnp.array([m['train_losses'] for m in metrics]), axis=0),
        'test_loss_mean': jnp.mean(jnp.array([m['test_losses'] for m in metrics]), axis=0),
        'test_loss_std': jnp.std(jnp.array([m['test_losses'] for m in metrics]), axis=0),
        'final_size_mean': jnp.mean(jnp.array([m['final_size'] for m in metrics])),
        'final_size_std': jnp.std(jnp.array([m['final_size'] for m in metrics]))
    }

def save_aggregated_metrics(metrics, args):
    agg_metrics = aggregate_runs(args.out_root)
    output_path = Path(args.out_root) / "aggregated"
    jnp.save(output_path / "metrics.npy", agg_metrics)

    # Also save human-readable version
    with open(output_path / "metrics.txt", "w") as f:
        f.write(f"Final Network Size: {agg_metrics['final_size_mean']:.2f} ± {agg_metrics['final_size_std']:.2f}\n")
        f.write(f"Final Test Loss: {agg_metrics['test_loss_mean'][-1]:.4f} ± {agg_metrics['test_loss_std'][-1]:.4f}")
