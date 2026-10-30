import torch
from torch import nn


class LSTM(nn.Module):
    """LSTM predictor"""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0.0,
        bidirectional=False,
        proj_size=0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            device=device,
            dtype=dtype,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        if hidden_state is None:
            hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )

        if cell_state is None:
            cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )

        out, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        out = self.fc(out)

        return out, hidden_state, cell_state
