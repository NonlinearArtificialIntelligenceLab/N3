import wandb
import jax.numpy as jnp
from beartype import beartype
from typing import Optional, Dict, Any, Union, Literal
from n3.architecture.model import ModelLike

class WandbLogger:
    """Centralized logging utility for Weights & Biases"""

    @beartype
    def __init__(self,
                project: str = "growing-networks",
                exp_name: str = "default",
                path_components: Dict[str, Any] = {},
                config: Optional[Dict[str, Any]] = None,
                enable: bool = True):
        if not enable:
            self.enabled = False
            return

        self.enabled = True
        run_name = self._generate_run_name(exp_name, path_components)
        tags = self._generate_tags(path_components)

        self.run = wandb.init(
            project=project,
            name=run_name,
            group=exp_name,
            tags=tags,
            config=config,
        )

    def _generate_run_name(self, exp_name: str,
                        components: Dict[str, Any]) -> str:
        """Create human-readable run name"""
        params = [
            f"{k[:3]}={v}"
            for k, v in components.items()
            if k in ['hidden_size', 'dataset_size', 'learning_rate']
        ]
        return f"{exp_name}_{'_'.join(params)}"

    def _generate_tags(self, components: Dict[str, Any]) -> list[str]:
        """Create searchable tags from parameters"""
        return [
            f"hs-{components.get('hidden_size', '')}",
            f"ds-{components.get('dataset_size', '')}",
            f"lr-{components.get('learning_rate', '')}",
            components.get('task_type', 'unknown')
        ]

    @beartype
    def watch_model(self,
                model: ModelLike,
                log: Literal['gradients', 'parameters', 'all'] = "gradients",
                log_freq: int = 100):
        """Track model parameters and gradients"""
        if self.enabled:
            wandb.watch(
                model,
                log=log,
                log_freq=log_freq,
                log_graph=(log == "parameters")
            )

    @beartype
    def log_metrics(self,
                metrics: Dict[str, Union[float, jnp.ndarray]],
                step: int):
        """Log training metrics"""
        if self.enabled:
            # Convert JAX arrays to native Python types
            metrics = {k: float(v) if isinstance(v, jnp.ndarray) else v
                    for k, v in metrics.items()}
            wandb.log(metrics, step=step)


    def __del__(self):
        if self.enabled:
            self.run.finish()
