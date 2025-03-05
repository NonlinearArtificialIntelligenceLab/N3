import argparse
import os
import json

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from sklearn.preprocessing import MinMaxScaler

from jaxtyping import Float, Array, Int

from n3.architecture.controller import IdentityController, ControllerLike
from n3.architecture.model import N3, ModelLike
from n3.utils.utils import grad_norm
from n3.utils.metrics import accuracy, cross_entropy, confusion_matrix
from n3.data import mnist1d
from n3.utils.wandb import WandbLogger
from n3.utils.path_manager import build_output_path

def argument_parser():
    parser = argparse.ArgumentParser(
        description="Runner for N3 classification on MNIST1D dataset."
    )
    parser.add_argument("--num_runs", type=int, default=1,
                    help="Number of independent runs to average over")
    parser.add_argument("--base_seed", type=int, default=0,
                    help="Base seed for random number generation")
    parser.add_argument("--n_samples", type=int, default=2**15,
                    help="Number of samples to generate")
    parser.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction of samples to use for testing")
    parser.add_argument("--N_max", type=int, default=10,
                    help="Per layer max number of neurons")
    parser.add_argument("--size_influence", type=float, default=0.32,
                    help="Influence of size loss")
    parser.add_argument("--epochs", type=int, default=5_000,
                    help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                    help="Learning rate for optimizer")
    parser.add_argument("--out_root", type=str, default="./output",
                    help="Root directory for all outputs")
    parser.add_argument("--exp_name", type=str, default="default",
                    help="Experiment name for organizing results")
    parser.add_argument("--log_every", type=int, default=100, help="log every n epochs")
    parser.add_argument("--wandb", action="store_true",
                    help="Enable Weights & Biases logging")
    parser.add_argument("--group", type=str, default=None,
                    help="W&B experiment group name")
    return parser

def compute_base_loss(
    model: ModelLike,
    control: ControllerLike,
    x: Float[Array, "batch 2"],
    y: Int[Array, "batch"],
) -> Float[Array, ""]:
    pred_y = jax.nn.log_softmax(jax.vmap(model, in_axes=(0, None))(x, control))
    loss = cross_entropy(y, pred_y)
    return loss


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
    x: Float[Array, "batch 2"],
    y: Int[Array, "batch"],
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
    x: Float[Array, "batch 2"],
    y: Int[Array, "batch"],
) -> Float[Array, ""]:
    return compute_base_loss(model, controller, x, y) + compute_size_loss(
        controller, size_influence
    )



def main():
    parser = argument_parser()
    args = parser.parse_args()

    args_dict = vars(args)
    args_dict['script'] = os.path.basename(__file__)

    for run_idx in range(args.num_runs):
        args_dict['run_idx'] = run_idx
        args.seed = args.base_seed + run_idx
        args_dict['seed'] = args.base_seed + run_idx

        output_path = build_output_path(args_dict)

        os.makedirs(output_path, exist_ok=True)

        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(args_dict, f)

        path_components = {
            'hidden_size': args.N_max,
            'dataset_size': args.n_samples,
            'learning_rate': args.learning_rate,
            'task_type': 'regression' if 'bessel' in __file__ else 'classification',
            'run_idx': run_idx,
            'seed': args_dict['seed']
        }

        # Initialize logger
        logger = WandbLogger(
            project="growing-nets",
            exp_name=args.exp_name,
            path_components=path_components,
            config=vars(args),
            enable=args.wandb
        )


        x_train, x_test, y_train, y_test = mnist1d.generate_data(
            n_samples=args.n_samples,
            test_size=args.test_size,
            scaler=MinMaxScaler(feature_range=(-1, 1)),
            seed=args.seed,
        )

        model_key, control_key = jax.random.split(jax.random.PRNGKey(args.seed))
        n3 = N3(40, 10, [args.N_max], model_key)  # 40 input features, 10 classes
        control = IdentityController(1, control_key)

        optim = optax.adam(learning_rate=args.learning_rate)
        opt_state = optim.init(eqx.filter([n3, control], eqx.is_inexact_array))

        # Training loop
        epoch_list = []
        test_losses = []
        test_accuracies = []
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
                test_accuracy = accuracy(n3, control, x_test, y_test)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
                train_losses.append(train_loss)
                controls.append(control.params.item() ** 2)
                control_grad_norms.append(
                    grad_norm(
                        eqx.filter_grad(compute_size_loss)(control, args.size_influence)
                    )
                )
                metrics = {
                            "train/loss": float(train_loss),
                            "test/loss": float(test_loss),
                            "test/accuracy": float(test_accuracy),
                            "network/size": control.params.item() ** 2,
                            "learning/control_grad_norm": float(control_grad_norms[-1])
                    }
                logger.log_metrics(metrics, epoch)

        # Save metrics
        cmat = confusion_matrix(n3, control, x_test, y_test)

        np.savetxt(f"{output_path}/epochs.txt", epoch_list)
        np.savetxt(f"{output_path}/test_losses.txt", test_losses)
        np.savetxt(f"{output_path}/test_accuracies.txt", test_accuracies)
        np.savetxt(f"{output_path}/train_losses.txt", train_losses)
        np.savetxt(f"{output_path}/controls.txt", controls)
        np.savetxt(f"{output_path}/control_grad_norms.txt", control_grad_norms)
        np.savetxt(f"{output_path}/confusion_matrix.txt", cmat)

        del logger


if __name__ == "__main__":
    main()
