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
                group: Optional[str] = None,
                config: Optional[Dict[str, Any]] = None,
                enable: bool = True):
        self.enabled = enable
        if self.enabled:
            self.run = wandb.init(
                project=project,
                group=group,
                config=config,
            )

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
