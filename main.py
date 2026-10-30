"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Chong Xin Le (S6865615)

Some utility functions like set_deterministic_behaviour and get_accelerator_device
reused from the 2 members, Kevin and Federico, their repository
for a project for the NLP course 2025-2026.
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lstm import LSTM
from preprocess import load_dataset
from utilities import get_accelerator_device, set_deterministic_behaviour

AMOUNT_FEATURES = 12

RANDOM_SEED = 42
AMOUNT_TRAINING_EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
HIDDEN_SIZE = 128


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

    trajectories = [trajectory[:min_length_trajectory] for trajectory in trajectories]
    trajectories = torch.tensor(trajectories, dtype=torch.float32)

    amount_trajectories = trajectories.size(0)

    train_fraction = 0.8
    slice_index = int(amount_trajectories * train_fraction)

    train_data = trajectories[0:slice_index]
    test_data = trajectories[slice_index:-1]

    X_train = train_data[:, :-1, :]
    Y_train = train_data[:, 1:, :]

    X_test = test_data[:, :-1, :]
    Y_test = test_data[:, 1:, :]

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    # shuffle must be false to prevent future leaking to past
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTM(
        input_size=AMOUNT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        output_size=AMOUNT_FEATURES,
        batch_first=True,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    for epoch in range(AMOUNT_TRAINING_EPOCHS):
        model.train()

        epoch_loss = 0.0

        hidden_state = None
        cell_state = None

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            if hidden_state is not None:
                hidden_state = hidden_state.detach()

            if cell_state is not None:
                cell_state = cell_state.detach()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions, hidden_state, cell_state = model(
                batch_x, hidden_state, cell_state
            )

            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == AMOUNT_TRAINING_EPOCHS - 1:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{AMOUNT_TRAINING_EPOCHS}], Loss: {avg_loss:.4f}")

    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    print(f"Done (execution duration: {time_elapsed})")


if __name__ == "__main__":
    main()
