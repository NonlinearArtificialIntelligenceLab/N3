import wandb
import time
import threading
import psutil
import jax.numpy as jnp
from beartype import beartype
from typing import Optional, Dict, Any, Union
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
                settings=wandb.Settings(
                    _disable_stats=True,
                    _disable_meta=True,
                    _service_wait=300
                )
            )
            self._start_system_monitor()

    @beartype
    def watch_model(self,
                model: ModelLike,
                log: str = "gradients",
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

    def _start_system_monitor(self):
        """Background thread for system metrics"""
        if self.enabled:
            self.monitor_running = True
            self.monitor_thread = threading.Thread(
                target=self._log_system_metrics,
                daemon=True
            )
            self.monitor_thread.start()

    def _log_system_metrics(self):
        """Log system resource usage"""
        while getattr(self, 'monitor_running', True):
            try:
                metrics = {
                    "system/cpu_usage": psutil.cpu_percent(),
                    "system/memory_usage": psutil.virtual_memory().percent,
                    "system/gpu_memory": self._get_gpu_memory()
                }
                self.log_metrics(metrics, step=wandb.run.step)
            except Exception as e:
                print(f"System monitoring error: {e}")
            time.sleep(10)

    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage percentage"""
        try:
            from jax.lib import xla_bridge
            device = xla_bridge.get_backend().devices()[0]
            return device.memory_percent()
        except:
            return 0.0

    def __del__(self):
        if self.enabled:
            self.monitor_running = False
            self.run.finish()
