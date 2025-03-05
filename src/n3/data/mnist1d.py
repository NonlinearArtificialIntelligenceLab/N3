import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import requests
import gzip
import os

from jaxtyping import Array, Float, Int

def download_mnist1d(data_dir: str = "../../DATA/mnist1d") -> tuple[np.ndarray, np.ndarray]:
    """Download MNIST1D dataset from GitHub."""
    os.makedirs(data_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/greydanus/mnist1d/master/data"
    files = ["X.npy.gz", "Y.npy.gz"]

    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "Y.npy")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        for file in files:
            url = f"{base_url}/{file}"
            response = requests.get(url)
            compressed_data = response.content

            # Decompress and save
            decompressed_data = gzip.decompress(compressed_data)
            output_path = os.path.join(data_dir, file.replace('.gz', ''))
            with open(output_path, 'wb') as f:
                f.write(decompressed_data)

    X = np.load(x_path)
    Y = np.load(y_path)
    return X, Y

def generate_data(
    n_samples: int = 1000,
    test_size: float = 0.2,
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1)),
    seed: int = 0,
) -> tuple[
    Float[Array, "train_samples 40"],
    Float[Array, "test_samples 40"],
    Int[Array, "train_samples"],
    Int[Array, "test_samples"],
]:
    """
    Generate MNIST1D dataset.

    Parameters:
    - n_samples: Number of samples to use (will be capped at available data)
    - test_size: Fraction of data to use for testing
    - scaler: Scaler to use for feature normalization
    - seed: Random seed

    Returns:
    - x_train: Training features
    - x_test: Test features
    - y_train: Training labels
    - y_test: Test labels
    """
    X, Y = download_mnist1d()

    # Limit to requested number of samples
    if n_samples < len(X):
        indices = np.random.RandomState(seed).choice(len(X), n_samples, replace=False)
        X = X[indices]
        Y = Y[indices]

    # Scale features
    X_scaled = scaler.fit_transform(X)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=test_size, random_state=seed
    )

    return (
        jnp.array(x_train),
        jnp.array(x_test),
        jnp.array(y_train),
        jnp.array(y_test)
    )