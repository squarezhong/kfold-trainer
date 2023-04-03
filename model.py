import torch.nn as nn

# simple fnn
class FNN(nn.Module):
    def __init__(self, input_size, n_classes) -> None:
        super().__init__()

        self.input_size = input_size
        self.n_classed = n_classes

        self.dense1 = nn.Linear(input_size, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 28)
        self.dense4 = nn.Linear(28, 10)
        self.dense5 = nn.Linear(10, n_classes)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.dense1(input))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.relu(self.dense4(x))
        return(self.dense5(x))
        #x = F.softmax(x, dim=-1)