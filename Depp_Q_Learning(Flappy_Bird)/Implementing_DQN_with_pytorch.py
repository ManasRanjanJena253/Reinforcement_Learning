# Importing dependencies
import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features = state_dim,
                             out_features = hidden_dim)
        self.fc2 = nn.Linear(in_features = 512,
                             out_features = 256)
        self.fc3 = nn.Linear(in_features = 256,
                             out_features = 128)
        self.fc4 = nn.Linear(in_features = 128,
                             out_features = action_dim)

    def forward(self, x):
        x = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return self.fc4(x)

if __name__ == '__main__' :
    state_dim = 12
    action_dim = 2
    model = DQN(state_dim = state_dim, action_dim = action_dim)

    # Testing the model output on dummy data
    state = torch.randn(1, state_dim)   # Creating dummy data.
    output = model(state)
    print(output)