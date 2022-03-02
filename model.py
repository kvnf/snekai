import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import(
    Tuple
)
from custom_types import (
    MemoryCell
)
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, in_size: int, inner_size: int, out_size: int) -> None:
        super(DQN, self).__init__()
        self.first_layer = nn.Linear(in_size, inner_size)
        self.hidden_layer = nn.Linear(inner_size, inner_size)
        self.output_layer = nn.Linear(inner_size, out_size)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


class Trainer(object):
    def __init__(self, model, learning_rate: float, decay_factor: float) -> None:
        self.model = model
        self.lr = learning_rate
        self.gamma = decay_factor
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train(self, mem_cell: MemoryCell) -> float:
        # get torch tensors
        state_prev = torch.tensor(
            mem_cell.state_prev, device=device, dtype=torch.float)
        state_next = torch.tensor(
            mem_cell.state_next, device=device, dtype=torch.float)
        reward = torch.tensor(
            mem_cell.reward, device=device, dtype=torch.float)
        action = torch.tensor(
            mem_cell.action, device=device, dtype=torch.long)
        # predict
        pred_q_val = self.model(state_prev)
        target_q_val = pred_q_val.detach().clone()

        new_q = reward
        if not mem_cell.game_over:
            # Bellman
            new_q = reward + self.gamma * torch.max(self.model(state_next))

        target_q_val[torch.argmax(action).tolist()] = new_q
        self.optimizer.zero_grad()
        loss = self.criterion(target_q_val, pred_q_val)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy()
