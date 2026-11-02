"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Chong Xin Le (S6865615)

Some utility functions like set_deterministic_behaviour and get_accelerator_device
reused from the 2 members, Kevin and Federico, their repository
for a project for the NLP course 2025-2026.
"""

import copy
import time

import numpy as np
import optuna
import torch
from optuna.trial import Trial
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lstm import LSTM
from preprocess import load_dataset
from utilities import get_accelerator_device, set_deterministic_behaviour

AMOUNT_FEATURES = 12

RANDOM_SEED = 42
AMOUNT_OPTUNA_TRIALS = 20

CRITERION = nn.MSELoss()


def objective(
    trial: Trial, device: str, X_train, Y_train, X_dev, Y_dev
) -> float:
    study = trial.study

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    amount_training_epochs = trial.suggest_int(
        "amount_training_epochs", 20, 100
    )
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])

    model_hyperparameters = {
        "input_size": AMOUNT_FEATURES,
        "hidden_size": hidden_size,
        "output_size": AMOUNT_FEATURES,
        "num_layers": 1,
        "batch_first": True,
    }

    model = LSTM(**model_hyperparameters).to(device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_dataset = TensorDataset(X_train, Y_train)
    dev_dataset = TensorDataset(X_dev, Y_dev)

    # shuffle must be false to prevent future leaking to past
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    patience = 5
    patience_counter = 0

    lowest_average_epoch_dev_loss = float("inf")

    for epoch in range(amount_training_epochs):
        model.train()

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions, _, _ = model(batch_x, None, None)

            loss = CRITERION(predictions, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_dev_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                predictions, _, _ = model(batch_x, None, None)

                dev_loss = CRITERION(predictions, batch_y)
                total_dev_loss += dev_loss.item()

        average_dev_loss = total_dev_loss / len(dev_loader)

        if average_dev_loss < lowest_average_epoch_dev_loss:
            lowest_average_epoch_dev_loss = average_dev_loss
            patience_counter = 0

            saved_best_checkpoint_info = study.user_attrs.get(
                "best_checkpoint_info"
            )
            if (
                saved_best_checkpoint_info is None
                or average_dev_loss < saved_best_checkpoint_info["loss"]
            ):
                checkpoint_info = {
                    "loss": average_dev_loss,
                    "hyperparameters": model_hyperparameters,
                    "state_dict": copy.deepcopy(model.state_dict()),
                }
                study.set_user_attr("best_checkpoint_info", checkpoint_info)
        else:
            patience_counter += 1

        trial.report(average_dev_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if patience_counter >= patience:
            break

    return lowest_average_epoch_dev_loss


def main():
    start_time = time.perf_counter()

    set_deterministic_behaviour(RANDOM_SEED)
    device = get_accelerator_device()

    trajectories, _, _ = load_dataset()

    min_length_trajectory = None
    trajectories_mismatch_found = False
    for trajectory in trajectories:
        length_trajectory = len(trajectory)
        if min_length_trajectory is None:
            min_length_trajectory = length_trajectory
            continue

        if (
            min_length_trajectory != length_trajectory
            and not trajectories_mismatch_found
        ):
            trajectories_mismatch_found = True
            print("-" * 20)
            print(
                f"trajectories not same size! min: {min_length_trajectory} current: {length_trajectory}. No other mismatches will be reported."
            )
            print("-" * 20)

        min_length_trajectory = min(min_length_trajectory, length_trajectory)

    trajectories = [
        trajectory[:min_length_trajectory] for trajectory in trajectories
    ]
    trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32)

    amount_trajectories = trajectories.size(0)

    train_fraction = 0.6
    dev_fraction = 0.2

    train_end_index = int(amount_trajectories * train_fraction)
    validate_end_index = train_end_index + int(
        amount_trajectories * dev_fraction
    )

    train_data = trajectories[0 : train_end_index + 1]
    dev_data = trajectories[train_end_index + 1 : validate_end_index + 1]
    test_data = trajectories[validate_end_index + 1 :]

    X_train = train_data[:, :-1, :]
    Y_train = train_data[:, 1:, :]

    X_dev = dev_data[:, :-1, :]
    Y_dev = dev_data[:, 1:, :]

    X_test = test_data[:, :-1, :]
    Y_test = test_data[:, 1:, :]

    test_dataset = TensorDataset(X_test, Y_test)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    study.optimize(
        lambda trial: objective(trial, device, X_train, Y_train, X_dev, Y_dev),
        n_trials=AMOUNT_OPTUNA_TRIALS,
    )

    best_checkpoint_info = study.user_attrs["best_checkpoint_info"]

    model = LSTM(**best_checkpoint_info["hyperparameters"]).to(device)
    model.load_state_dict(best_checkpoint_info["state_dict"])
    model.eval()

    print("best model params:")
    print(study.best_params)
    print(best_checkpoint_info)

    # shuffle must be false to prevent future leaking to past
    test_batch_size = 64
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )

    baseline_model = LinearRegression()
    baseline_model.fit(X_train[0], Y_train[0])

    prediction = baseline_model.predict(X_test[0])
    mse = mean_squared_error(Y_test[0][:12][0], prediction[0])
    print("mse: ", mse)

    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    print(f"Done (execution duration: {time_elapsed})")


if __name__ == "__main__":
    main()
