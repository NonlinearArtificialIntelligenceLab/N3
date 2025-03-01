import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def gaussian_mixture_function(
    x: np.ndarray,
    centers: np.ndarray = np.array([-2.0, 0.0, 2.0]),
    widths: np.ndarray = np.array([0.5, 0.8, 0.3]),
    weights: np.ndarray = np.array([1.0, 1.5, 0.8]),
) -> np.ndarray:
    """Mixture of Gaussians with different centers, widths, and weights."""
    y = np.zeros_like(x)
    for c, w, a in zip(centers, widths, weights):
        y += a * np.exp(-((x - c) ** 2) / (2 * w**2))
    return y


def generate_data(
    n_samples: int = 1000,
    test_size: float = 0.2,
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1)),
    noise_level: float = 0.1,
    seed: int = 0,
):
    np.random.seed(seed)
    x = np.linspace(-4, 4, n_samples).reshape(-1, 1)
    y = gaussian_mixture_function(x)
    y += np.random.normal(0, noise_level, size=y.shape)

    x_scaled = scaler.fit_transform(x)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_scaled, test_size=test_size, random_state=seed
    )

    return jnp.array(x_train), jnp.array(x_test), jnp.array(y_train), jnp.array(y_test)
