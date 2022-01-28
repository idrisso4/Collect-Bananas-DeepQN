import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dense_1 = nn.Linear(state_size, 64)
        self.dense_2 = nn.Linear(64, 128)
        self.dense_3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.dense_1(state))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))

        return self.output(x)
