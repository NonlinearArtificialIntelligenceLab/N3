from dataclasses import dataclass
from typing import Tuple

@dataclass
class ScalingConfig:
    # Model parameters
    hidden_sizes: tuple[int, ...] = (64, 128, 256, 512)
    size_influence: float = 0.32

    # Data parameters
    dataset_sizes: tuple[int, ...] = (1000, 10000, 100000)
    num_classes: tuple[int, ...] = (3, 5, 7)

    # Training parameters
    batch_sizes: tuple[int, ...] = (64, 128, 256)
    learning_rates: tuple[float, ...] = (1e-4, 3e-4, 1e-3)

    # Resource management
    max_gpu_mem_util: float = 0.8  # Target GPU memory utilization
    cpu_cores_per_task: int = 2     # For CPU-bound preprocessing

    def generate_sweep(self):
        """Yield configurations for parameter sweep"""
        from itertools import product
        return product(
            self.hidden_sizes,
            self.dataset_sizes,
            self.num_classes,
            self.batch_sizes,
            self.learning_rates
        )
