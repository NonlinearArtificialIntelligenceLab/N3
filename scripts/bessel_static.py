import argparse
import logging
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from sklearn.preprocessing import MinMaxScaler

from jaxtyping import Float, Array


from n3.architecture.controller import IdentityController, ControllerLike
from n3.architecture.model import N3, ModelLike
from n3.data import bessel
from n3.utils.utils import grad_norm


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Runner for N3 regression on Bessel function dataset."
    )
    parser.add_argument(
        "--n_samples", type=int, default=2**15, help="Number of samples to generate"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of samples to use for testing",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for random number generator"
    )
    parser.add_argument(
        "--N_max", type=int, default=10, help="Per layer max number of neurons"
    )
    parser.add_argument(
        "--size_influence", type=float, default=0.32, help="Influence of size loss"
    )
    parser.add_argument(
        "--epochs", type=int, default=5_000, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--out_path", type=str, default="../output/test/", help="Path to save metrics"
    )
    parser.add_argument("--log_every", type=int, default=100, help="log every n epochs")
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging verbosity",
    )
    parser.add_argument("--console", action="store_true", help="Log to console")
    return parser


def compute_base_loss(
    model: ModelLike,
    control: ControllerLike,
    x: Float[Array, "batch 1"],
    y: Float[Array, "batch 1"],
) -> Float[Array, ""]:
    pred = jax.vmap(model, in_axes=(0, None))(x, control)
    return jnp.mean((pred - y) ** 2)


def compute_size_loss(
    controller: ControllerLike, size_influence: float
) -> Float[Array, ""]:
    N = controller(jnp.ones((1,)))
    return size_influence * jnp.mean((N - 1.0) ** 2)


@eqx.filter_jit
def make_step(
    model: ModelLike,
    controller: ControllerLike,
    size_influence: float,
    x: Float[Array, "batch 1"],
    y: Float[Array, "batch 1"],
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[Float[Array, ""], ModelLike, ControllerLike, optax.OptState]:
    loss_base, grads_base = eqx.filter_value_and_grad(compute_base_loss)(
        model, controller, x, y
    )
    loss_size, grads_size = eqx.filter_value_and_grad(compute_size_loss)(
        controller, size_influence
    )
    loss = loss_base + loss_size

    updates, opt_state = optim.update([grads_base, grads_size], opt_state)

    model = eqx.apply_updates(model, updates[0])  # type: ignore
    controller = eqx.apply_updates(controller, updates[1])  # type: ignore
    return loss, model, controller, opt_state


@eqx.filter_jit
def test_step(
    model: ModelLike,
    controller: ControllerLike,
    size_influence: float,
    x: Float[Array, "batch 1"],
    y: Float[Array, "batch 1"],
) -> Float[Array, ""]:
    return compute_base_loss(model, controller, x, y) + compute_size_loss(
        controller, size_influence
    )


def main():
    parser = argument_parser()
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.verbosity),
        filename=f"{args.out_path}info.log",
        filemode="w",
    )
    logger = logging.getLogger(__name__)
    if args.console:
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(console_handler)

    # Dataset
    x_train, x_test, y_train, y_test = bessel.generate_data(
        n_samples=args.n_samples,
        test_size=args.test_size,
        scaler=MinMaxScaler(feature_range=(-1, 1)),
        seed=args.seed,
    )

    # Model and Controller
    model_key, control_key = jax.random.split(jax.random.PRNGKey(args.seed))
    n3 = N3(1, 1, [args.N_max], model_key)
    control = IdentityController(1, control_key)  # this line defines the static nature

    optim = optax.adam(learning_rate=args.learning_rate)
    opt_state = optim.init(eqx.filter([n3, control], eqx.is_inexact_array))

    # Training loop
    epoch_list = []
    test_losses = []
    train_losses = []
    controls = []
    control_grad_norms = []

    for epoch in range(args.epochs):
        train_loss, n3, control, opt_state = make_step(
            n3, control, args.size_influence, x_train, y_train, optim, opt_state
        )

        if epoch % args.log_every == 0:
            epoch_list.append(epoch)
            test_loss = test_step(n3, control, args.size_influence, x_test, y_test)

            test_losses.append(test_loss)
            train_losses.append(train_loss)
            controls.append(control.params.item() ** 2)
            control_grad_norms.append(
                grad_norm(
                    eqx.filter_grad(compute_size_loss)(control, args.size_influence)
                )
            )
            logger.info(
                f"epoch: {epoch_list[-1]}, train_loss: {train_losses[-1]:.4e}, test_loss: {test_losses[-1]:.4e} control2: {controls[-1]:.4e}"
            )
            logger.info(f"Control_grad_norm: {control_grad_norms[-1]:.4e}")

    # Save metrics
    np.savetxt(f"{args.out_path}epochs.txt", epoch_list)
    np.savetxt(f"{args.out_path}test_losses.txt", test_losses)
    np.savetxt(f"{args.out_path}train_losses.txt", train_losses)
    np.savetxt(f"{args.out_path}controls.txt", controls)
    np.savetxt(f"{args.out_path}control_grad_norms.txt", control_grad_norms)


if __name__ == "__main__":
    main()